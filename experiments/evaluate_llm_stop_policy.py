"""Evaluate a frozen LLM self-judgment STOP/CONTINUE baseline.

The evaluator reuses fixed retrieval trajectories and frozen-reader answer
scores.  At each non-terminal state, Qwen sees only the question, the evidence
that the reader would see, and previous search queries.  It never sees gold
answers, supporting facts, hop counts, EM, or F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.evaluate_rule_stop_policy import _paired_bootstrap
from experiments.evaluate_stop_answer_oracle import _order_sources_for_reader, _restore_sources
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from src.document_loader import Chunk
from src.llm_stop_judge import (
    PROMPT_VERSION,
    FrozenOllamaStopJudge,
    StopJudgeCache,
    StopJudgeConfig,
)
from src.qa_evaluation import evidence_context, ollama_model_identity
from src.settings import settings


FORMAT_VERSION = 1
POLICY_NAME = "qwen3_8b_zero_shot_evidence_sufficiency_v1"
ACTIONS = ("vector", "planned_continue_1", "planned_continue_2", "planned_continue_3")
MAX_DECISION_STEPS = len(ACTIONS) - 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen Qwen self-judgment STOP baseline")
    parser.add_argument(
        "--answer-root",
        default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_validation_2000",
    )
    parser.add_argument(
        "--trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument(
        "--index-root",
        default="data/benchmarks/multihop/fixed_corpus/index/validation",
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--max-rows-per-dataset", type=int, default=2000)
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--context-allocation", choices=("sequential", "balanced"), default="sequential")
    parser.add_argument("--context-source-order", choices=("trajectory", "initial_first"), default="trajectory")
    parser.add_argument("--context-compression", choices=("none", "query_focused"), default="none")
    parser.add_argument("--focused-sentences-per-source", type=int, default=2)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--judge-model", default=settings.ollama_model)
    parser.add_argument("--judge-max-tokens", type=int, default=120)
    parser.add_argument("--judge-timeout", type=int, default=300)
    parser.add_argument("--judge-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        default="data/benchmarks/multihop/llm_policy/qwen3_8b_self_judge_validation_2000",
    )
    parser.add_argument(
        "--summary-out",
        default="data/benchmarks/multihop/llm_policy/qwen3_8b_self_judge_validation_2000.json",
    )
    parser.add_argument(
        "--predictions-out",
        default="data/benchmarks/multihop/llm_policy/qwen3_8b_self_judge_validation_2000_predictions.jsonl",
    )
    parser.add_argument(
        "--cache-path",
        default="data/benchmarks/multihop/llm_policy/llm_stop_judge_cache.sqlite3",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_rows_per_dataset < 1 or args.context_character_budget < 1:
        parser.error("행 수와 context budget은 1 이상이어야 합니다.")
    if args.judge_max_tokens < 1 or args.judge_timeout < 1 or args.log_every < 1:
        parser.error("judge token/timeout/log 간격은 1 이상이어야 합니다.")
    return args


def _load_inputs(args, dataset: str) -> tuple[list[dict], dict[str, dict], dict[str, Chunk], dict[str, list[Chunk]]]:
    answer_path = Path(args.answer_root) / f"{dataset}_{args.split}.jsonl"
    trajectory_path = Path(args.trajectory_root) / f"{dataset}_{args.split}.jsonl"
    chunk_path = Path(args.index_root) / dataset / "chunks.json"
    for path in (answer_path, trajectory_path, chunk_path):
        if not path.exists():
            raise SystemExit(f"필요한 입력 파일이 없습니다: {path}")

    rows = list(_iter_jsonl(answer_path))[: args.max_rows_per_dataset]
    selected_ids = {str(row["id"]) for row in rows}
    trajectories = {
        str(row["id"]): row
        for row in _iter_jsonl(trajectory_path)
        if str(row["id"]) in selected_ids
    }
    if len(rows) != len(trajectories):
        raise SystemExit(f"{dataset}: answer/trajectory ID가 일치하지 않습니다.")
    for row in rows:
        if tuple(row.get("actions", {})) != ACTIONS:
            raise SystemExit(f"{dataset}: 지원하지 않는 answer action입니다: {row.get('id')}")
        trajectory = trajectories[str(row["id"])]
        if tuple(trajectory.get("actions", {})) != ACTIONS:
            raise SystemExit(f"{dataset}: 지원하지 않는 trajectory action입니다: {row.get('id')}")

    chunks = [Chunk(**item) for item in json.loads(chunk_path.read_text(encoding="utf-8"))]
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunks_by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_source.setdefault(chunk.source, []).append(chunk)
    return rows, trajectories, chunks_by_id, chunks_by_source


def _input_signature(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_config(args, dataset: str, judge_config: StopJudgeConfig) -> dict:
    answer_path = Path(args.answer_root) / f"{dataset}_{args.split}.jsonl"
    trajectory_path = Path(args.trajectory_root) / f"{dataset}_{args.split}.jsonl"
    chunk_path = Path(args.index_root) / dataset / "chunks.json"
    return {
        "format_version": FORMAT_VERSION,
        "experiment": "frozen_llm_self_judgment_stop_policy",
        "policy": POLICY_NAME,
        "dataset": dataset,
        "split": args.split,
        "maximum_rows": args.max_rows_per_dataset,
        "actions": list(ACTIONS),
        "answer_signature": _input_signature(answer_path),
        "trajectory_signature": _input_signature(trajectory_path),
        "chunk_signature": _input_signature(chunk_path),
        "context_character_budget": args.context_character_budget,
        "context_allocation": args.context_allocation,
        "context_source_order": args.context_source_order,
        "context_compression": args.context_compression,
        "focused_sentences_per_source": args.focused_sentences_per_source,
        "judge": asdict(judge_config),
        "gold_inputs_exposed_to_policy": [],
    }


def _manifest_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    actual = json.loads(path.read_text(encoding="utf-8"))
    return all(actual.get(key) == value for key, value in expected.items())


def _write_manifest(path: Path, config: dict, status: str, completed: int, requested: int) -> None:
    payload = {**config, "status": status, "completed_rows": completed, "requested_rows": requested}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _decision_context(
    trajectory: dict,
    step: int,
    chunks_by_id: dict[str, Chunk],
    chunks_by_source: dict[str, list[Chunk]],
    args,
) -> tuple[str, tuple[dict, ...], tuple[str, ...]]:
    initial_sources = _restore_sources(trajectory["actions"][ACTIONS[0]], chunks_by_id, chunks_by_source)
    initial_ids = tuple(chunk.chunk_id for chunk, _ in initial_sources)
    sources = _restore_sources(trajectory["actions"][ACTIONS[step]], chunks_by_id, chunks_by_source)
    sources = _order_sources_for_reader(sources, initial_ids, args.context_source_order)
    previous_queries = tuple(
        str(item.get("query", "")).strip()
        for item in trajectory.get("planner_audit", [])[:step]
        if str(item.get("query", "")).strip()
    )
    context, provenance = evidence_context(
        sources,
        args.context_character_budget,
        allocation=args.context_allocation,
        compression=args.context_compression,
        focus_queries=previous_queries,
        focused_sentences_per_source=args.focused_sentences_per_source,
    )
    return context, provenance, previous_queries


def _evaluate_row(
    judge: FrozenOllamaStopJudge,
    row: dict,
    trajectory: dict,
    chunks_by_id: dict[str, Chunk],
    chunks_by_source: dict[str, list[Chunk]],
    args,
) -> dict:
    decisions = []
    stop_step = MAX_DECISION_STEPS
    for step in range(MAX_DECISION_STEPS):
        context, provenance, previous_queries = _decision_context(
            trajectory, step, chunks_by_id, chunks_by_source, args
        )
        result = judge.judge(str(row["question"]), context, previous_queries)
        decisions.append(
            {
                "state_step": step,
                "action": ACTIONS[step],
                "decision": result.decision,
                "supported_answer": result.supported_answer,
                "missing_information": result.missing_information,
                "confidence": result.confidence,
                "valid": result.valid,
                "cache_hit": result.cache_hit,
                "latency_seconds": result.latency_seconds,
                "prompt_eval_count": result.prompt_eval_count,
                "eval_count": result.eval_count,
                "context_characters": len(context),
                "visible_evidence_content_ids": [str(item["content_id"]) for item in provenance],
            }
        )
        if result.decision == "STOP":
            stop_step = step
            break

    selected_action = ACTIONS[stop_step]
    selected = row["actions"][selected_action]
    planner_latency = sum(
        float(item.get("latency_seconds", 0.0))
        for item in row.get("planner_audit", [])[:stop_step]
    )
    return {
        "format_version": FORMAT_VERSION,
        "policy": POLICY_NAME,
        "id": str(row["id"]),
        "dataset": str(row["dataset"]),
        "selected_action": selected_action,
        "stop_step": stop_step,
        "decision_trace": decisions,
        "direct_em": float(row["direct_em"]),
        "direct_f1": float(row["direct_f1"]),
        "policy_em": float(selected["em"]),
        "policy_f1": float(selected["f1"]),
        "source_recall": float(selected["source_recall"]),
        "all_support_recall": float(selected["all_support_recall"]),
        "context_characters": int(selected.get("context_characters", 0)),
        "reader_latency_seconds": float(selected.get("reader_latency_seconds", 0.0)),
        "planner_latency_seconds": planner_latency,
        "judge_calls": len(decisions),
        "judge_latency_seconds": float(sum(item["latency_seconds"] for item in decisions)),
        "judge_prompt_tokens": int(sum(item["prompt_eval_count"] or 0 for item in decisions)),
        "judge_output_tokens": int(sum(item["eval_count"] or 0 for item in decisions)),
        "invalid_judgments": int(sum(not item["valid"] for item in decisions)),
        "oracle_f1": float(row["oracle_f1"]),
    }


def _evaluate_dataset(judge, rows, trajectories, chunks_by_id, chunks_by_source, output_path, config, args):
    manifest_path = output_path.with_suffix(".manifest.json")
    cached: dict[str, dict] = {}
    if output_path.exists() or manifest_path.exists():
        if args.overwrite:
            cached = {}
        elif _manifest_matches(manifest_path, config):
            cached = {str(row["id"]): row for row in _iter_jsonl(output_path)} if output_path.exists() else {}
        else:
            raise SystemExit(
                f"기존 LLM STOP 결과 설정과 다릅니다: {output_path}\n"
                "다른 --output-root를 사용하거나 --overwrite를 지정하세요."
            )
    expected_ids = {str(row["id"]) for row in rows}
    rows_by_id = {row_id: row for row_id, row in cached.items() if row_id in expected_ids}
    if len(rows_by_id) == len(rows):
        _log(f"{config['dataset']}: LLM STOP {len(rows):,}개 전체 재사용")
        return [rows_by_id[str(row["id"])] for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output_path.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    _write_manifest(manifest_path, config, "in_progress", len(rows_by_id), len(rows))
    started = perf_counter()
    newly_completed = 0
    with output_path.open(mode, encoding="utf-8") as destination:
        for source_row in rows:
            row_id = str(source_row["id"])
            if row_id in rows_by_id:
                continue
            result = _evaluate_row(
                judge,
                source_row,
                trajectories[row_id],
                chunks_by_id,
                chunks_by_source,
                args,
            )
            destination.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows_by_id[row_id] = result
            newly_completed += 1
            completed = len(rows_by_id)
            if completed % args.log_every == 0 or completed == len(rows):
                elapsed = perf_counter() - started
                eta = elapsed / max(1, newly_completed) * (len(rows) - completed)
                current = list(rows_by_id.values())
                _log(
                    f"{config['dataset']} LLM STOP {completed:,}/{len(rows):,} "
                    f"({completed / len(rows):.1%}) | ETA {_duration(eta)} "
                    f"| EM/F1={np.mean([item['policy_em'] for item in current]):.3f}/"
                    f"{np.mean([item['policy_f1'] for item in current]):.3f} "
                    f"| 추가검색={np.mean([item['stop_step'] for item in current]):.3f} "
                    f"| judge호출={np.mean([item['judge_calls'] for item in current]):.3f}"
                )
                _write_manifest(manifest_path, config, "in_progress", completed, len(rows))
    _write_manifest(manifest_path, config, "ready", len(rows_by_id), len(rows))
    return [rows_by_id[str(row["id"])] for row in rows]


def _summary(records: list[dict], datasets: tuple[str, ...]) -> dict:
    summary = {}
    for dataset in datasets:
        selected = [row for row in records if row["dataset"] == dataset]
        step_counts = Counter(int(row["stop_step"]) for row in selected)
        direct_em = float(np.mean([row["direct_em"] for row in selected]))
        direct_f1 = float(np.mean([row["direct_f1"] for row in selected]))
        policy_em = float(np.mean([row["policy_em"] for row in selected]))
        policy_f1 = float(np.mean([row["policy_f1"] for row in selected]))
        decision_count = sum(int(row["judge_calls"]) for row in selected)
        summary[dataset] = {
            "n": len(selected),
            "direct": {"em": direct_em, "f1": direct_f1},
            "policy": {"em": policy_em, "f1": policy_f1},
            "gain": {"em": policy_em - direct_em, "f1": policy_f1 - direct_f1},
            "average_extra_searches": float(np.mean([row["stop_step"] for row in selected])),
            "average_total_searches": 1.0 + float(np.mean([row["stop_step"] for row in selected])),
            "average_judge_calls": float(np.mean([row["judge_calls"] for row in selected])),
            "average_judge_latency_seconds": float(np.mean([row["judge_latency_seconds"] for row in selected])),
            "average_judge_prompt_tokens": float(np.mean([row["judge_prompt_tokens"] for row in selected])),
            "average_judge_output_tokens": float(np.mean([row["judge_output_tokens"] for row in selected])),
            "invalid_judgment_rate": float(
                sum(int(row["invalid_judgments"]) for row in selected) / max(1, decision_count)
            ),
            "stop_steps": {
                str(step): {
                    "count": int(step_counts.get(step, 0)),
                    "rate": float(step_counts.get(step, 0) / len(selected)),
                }
                for step in range(len(ACTIONS))
            },
            "source_recall": float(np.mean([row["source_recall"] for row in selected])),
            "all_support_recall": float(np.mean([row["all_support_recall"] for row in selected])),
            "oracle_f1": float(np.mean([row["oracle_f1"] for row in selected])),
            "oracle_regret_f1": float(np.mean([row["oracle_f1"] - row["policy_f1"] for row in selected])),
        }
    return summary


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in sorted(records, key=lambda item: (item["dataset"], item["id"])):
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    started = perf_counter()
    session = requests.Session()
    try:
        resolved_model, model_digest = ollama_model_identity(
            args.ollama_host, args.judge_model, session=session
        )
    except RuntimeError as error:
        raise SystemExit(f"{error}\nOllama 실행 상태와 judge 모델을 확인하세요.") from error
    judge_config = StopJudgeConfig(
        host=args.ollama_host,
        model=resolved_model,
        model_digest=model_digest,
        max_tokens=args.judge_max_tokens,
        timeout_seconds=args.judge_timeout,
        seed=args.judge_seed,
        temperature=0.0,
        keep_alive=args.ollama_keep_alive,
        thinking=False,
    )
    _log(
        f"LLM self-judgment 시작: model={resolved_model}, digest={model_digest or '미제공'}, "
        f"temperature=0, thinking=false, prompt={PROMPT_VERSION}"
    )
    all_records = []
    output_root = Path(args.output_root)
    with StopJudgeCache(Path(args.cache_path)) as cache:
        judge = FrozenOllamaStopJudge(judge_config, cache, session=session)
        for dataset in args.datasets:
            _log(f"{dataset}: 순차 평가를 시작합니다.")
            rows, trajectories, chunks_by_id, chunks_by_source = _load_inputs(args, dataset)
            config = _manifest_config(args, dataset, judge_config)
            records = _evaluate_dataset(
                judge,
                rows,
                trajectories,
                chunks_by_id,
                chunks_by_source,
                output_root / f"{dataset}_{args.split}.jsonl",
                config,
                args,
            )
            all_records.extend(records)
            del chunks_by_id, chunks_by_source, trajectories, rows
    session.close()

    summary = {
        "format_version": FORMAT_VERSION,
        "policy": POLICY_NAME,
        "split": args.split,
        "judge": asdict(judge_config),
        "datasets": _summary(all_records, args.datasets),
        "paired_bootstrap_vs_direct": _paired_bootstrap(
            all_records, args.datasets, args.bootstrap_samples, args.bootstrap_seed
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "elapsed_seconds": perf_counter() - started,
    }
    _write_jsonl(Path(args.predictions_out), all_records)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(summary_path)
    for dataset, values in summary["datasets"].items():
        _log(
            f"{dataset}: Direct→LLM EM {values['direct']['em']:.3f}→{values['policy']['em']:.3f}, "
            f"F1 {values['direct']['f1']:.3f}→{values['policy']['f1']:.3f}, "
            f"평균 추가검색 {values['average_extra_searches']:.3f}, "
            f"judge 호출 {values['average_judge_calls']:.3f}"
        )
    _log(f"LLM self-judgment 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
