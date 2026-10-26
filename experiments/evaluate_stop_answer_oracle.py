"""Score every 4-hop STOP prefix with one frozen QA reader.

Input rows come from ``evaluate_iterative_hop_retrieval`` format v5.  Retrieval
has already generated one Direct state and three CONTINUE prefixes.  This
command changes no retrieval result and trains no policy; it only assigns
Answer EM/F1 utility to each possible STOP point and creates offline
STOP/CONTINUE labels for the next policy-training stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.evaluate_answer_oracle import _load_fixed_retriever, _score_sources
from experiments.train_multihop_policy import (
    SUPPORTED_DATASETS,
    _corpus_fingerprint,
    _dataset_seed,
    _duration,
    _iter_jsonl,
    load_processed_examples,
)
from src.qa_evaluation import (
    PROMPT_VERSION,
    SUPPORTED_PROMPT_VERSIONS,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    ollama_model_identity,
)
from src.settings import settings


FORMAT_VERSION = 2
DIRECT_ACTION = "vector"


def main() -> None:
    args = _arguments()
    started = perf_counter()
    session = requests.Session()
    try:
        resolved_model, model_digest = ollama_model_identity(
            args.ollama_host,
            args.reader_model,
            session=session,
        )
    except RuntimeError as error:
        raise SystemExit(f"{error}\nOllama 실행 상태와 reader 모델을 확인하세요.") from error
    reader_config = ReaderConfig(
        host=args.ollama_host,
        model=resolved_model,
        model_digest=model_digest,
        max_tokens=args.reader_max_tokens,
        timeout_seconds=args.reader_timeout,
        seed=args.reader_seed,
        temperature=0.0,
        keep_alive=args.ollama_keep_alive,
        prompt_version=args.reader_prompt_version,
        thinking=args.reader_thinking,
    )
    _log("4-hop STOP-prefix Answer Oracle 평가를 시작합니다. Reader와 검색 결과는 동결됩니다.")
    _log(
        f"Reader: model={resolved_model}, digest={model_digest or '미제공'}, "
        f"temperature=0, thinking={reader_config.thinking}, prompt={reader_config.prompt_version}"
    )

    all_rows: list[dict] = []
    shared_embedding_model = None
    shared_reranker = None
    with AnswerCache(Path(args.cache_path)) as cache:
        reader = FrozenOllamaReader(reader_config, cache, session=session)
        for dataset in args.datasets:
            trajectory_path = Path(args.trajectory_root) / f"{dataset}_{args.split}.jsonl"
            trajectory_manifest_path = trajectory_path.with_suffix(".manifest.json")
            if not trajectory_path.exists() or not trajectory_manifest_path.exists():
                raise SystemExit(f"trajectory 결과가 없습니다: {trajectory_path}")
            trajectory_manifest = json.loads(trajectory_manifest_path.read_text(encoding="utf-8"))
            if int(trajectory_manifest.get("max_evidence_hops", 0)) != 4:
                raise SystemExit(f"{dataset}: 최대 4-hop trajectory가 아닙니다: {trajectory_manifest_path}")
            trajectories = list(_iter_jsonl(trajectory_path))
            if args.max_rows_per_dataset:
                trajectories = trajectories[: args.max_rows_per_dataset]
            if not trajectories:
                raise SystemExit(f"trajectory 행이 없습니다: {trajectory_path}")

            processed_path = Path(args.processed_root) / dataset / f"{args.split}.jsonl"
            corpus_examples = load_processed_examples(
                processed_path,
                args.corpus_examples_per_dataset,
                _dataset_seed(args.corpus_seed, dataset),
            )
            example_by_id = {example.policy_id: example for example in corpus_examples}
            missing = {str(row["id"]) for row in trajectories} - set(example_by_id)
            if missing:
                raise SystemExit(f"{dataset}: 전처리 cohort에 trajectory ID가 없습니다: {next(iter(missing))}")
            corpus_fingerprint = _corpus_fingerprint(corpus_examples)
            if corpus_fingerprint != trajectory_manifest.get("corpus_fingerprint"):
                raise SystemExit(f"{dataset}: trajectory와 processed corpus fingerprint가 다릅니다.")
            retriever = _load_fixed_retriever(
                Path(args.index_root) / args.split / dataset,
                corpus_fingerprint,
                embedding_model=shared_embedding_model,
                reranker=shared_reranker,
            )
            shared_embedding_model = retriever.model or shared_embedding_model
            shared_reranker = retriever.reranker or shared_reranker
            chunks_by_id = {chunk.chunk_id: chunk for chunk in retriever.chunks}
            chunks_by_source: dict[str, list] = {}
            for chunk in retriever.chunks:
                chunks_by_source.setdefault(chunk.source, []).append(chunk)

            output_path = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
            trajectory_actions = _trajectory_actions(trajectories[0])
            actions = tuple(
                action
                for action in trajectory_actions
                if not args.score_actions or action in args.score_actions
            )
            manifest_config = {
                "format_version": FORMAT_VERSION,
                "experiment": "stop_prefix_answer_oracle_v2_context_allocation",
                "dataset": dataset,
                "split": args.split,
                "trajectory_signature": _trajectory_signature(trajectory_manifest),
                "corpus_fingerprint": corpus_fingerprint,
                "corpus_examples_per_dataset": args.corpus_examples_per_dataset,
                "corpus_seed": args.corpus_seed,
                "actions": list(actions),
                "max_evidence_hops": 4,
                "context_character_budget": args.context_character_budget,
                "context_allocation": args.context_allocation,
                "context_source_order": args.context_source_order,
                "context_compression": args.context_compression,
                "focused_sentences_per_source": args.focused_sentences_per_source,
                "reader": asdict(reader_config),
            }
            rows = _evaluate_dataset(
                reader,
                trajectories,
                example_by_id,
                chunks_by_id,
                chunks_by_source,
                output_path,
                manifest_config,
                actions=actions,
                context_character_budget=args.context_character_budget,
                context_allocation=args.context_allocation,
                context_source_order=args.context_source_order,
                context_compression=args.context_compression,
                focused_sentences_per_source=args.focused_sentences_per_source,
                log_every=args.log_every,
                overwrite=args.overwrite,
            )
            all_rows.extend(rows)
            _report(rows, dataset, actions)
    if len(args.datasets) > 1:
        _report_macro(all_rows, args.datasets, actions)
    session.close()
    _log(f"STOP-prefix Answer Oracle 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4-hop STOP prefix frozen-reader Answer EM/F1 평가")
    parser.add_argument("--trajectory-root", default="data/benchmarks/multihop/sequential_search_v5_4hop")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--output-root", default="data/benchmarks/multihop/stop_answer_oracle_v1")
    parser.add_argument("--cache-path", default="data/benchmarks/multihop/answer_cache.sqlite3")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument(
        "--score-actions",
        nargs="+",
        choices=(DIRECT_ACTION, "planned_continue_1", "planned_continue_2", "planned_continue_3"),
        help="지정한 STOP prefix만 reader로 평가합니다. 기본값은 네 prefix 전체입니다.",
    )
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument(
        "--context-allocation",
        choices=("sequential", "balanced"),
        default="sequential",
        help="sequential은 앞 문서 우선, balanced는 모든 문서에 입력 길이를 균등 배분",
    )
    parser.add_argument(
        "--context-source-order",
        choices=("trajectory", "initial_first"),
        default="trajectory",
        help="initial_first는 최초 검색 근거의 상대 순서를 보존하고 추가 근거를 뒤에 배치",
    )
    parser.add_argument(
        "--context-compression",
        choices=("none", "query_focused"),
        default="none",
        help="query_focused는 질문/검색어 관련 문장을 문서별로 우선 선택",
    )
    parser.add_argument("--focused-sentences-per-source", type=int, default=2)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--reader-model", default=settings.ollama_model)
    parser.add_argument("--reader-max-tokens", type=int, default=64)
    parser.add_argument("--reader-timeout", type=int, default=300)
    parser.add_argument("--reader-seed", type=int, default=42)
    parser.add_argument(
        "--reader-prompt-version",
        choices=SUPPORTED_PROMPT_VERSIONS,
        default=PROMPT_VERSION,
        help="동결 reader prompt 버전. evidence-chain은 답 전에 짧은 근거 사슬을 구조화합니다.",
    )
    parser.add_argument(
        "--reader-thinking",
        action="store_true",
        help="Qwen3 내부 reasoning을 활성화합니다. num_predict 예산을 함께 늘리는 것이 안전합니다.",
    )
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.corpus_examples_per_dataset < 1 or args.max_rows_per_dataset < 0:
        parser.error("corpus는 1 이상, max rows는 0 이상이어야 합니다.")
    if args.context_character_budget < 1 or args.reader_max_tokens < 1 or args.reader_timeout < 1:
        parser.error("context/token/timeout은 1 이상이어야 합니다.")
    if args.focused_sentences_per_source < 1:
        parser.error("focused sentences는 1 이상이어야 합니다.")
    if args.log_every < 1:
        parser.error("log 간격은 1 이상이어야 합니다.")
    return args


def _trajectory_actions(row: Mapping) -> tuple[str, ...]:
    actions = tuple(
        action
        for action in row.get("actions", {})
        if action == DIRECT_ACTION or action.startswith("planned_continue_")
    )
    expected = (DIRECT_ACTION, "planned_continue_1", "planned_continue_2", "planned_continue_3")
    if actions != expected:
        raise SystemExit(f"4-hop planned trajectory 행동이 필요합니다: expected={expected}, actual={actions}")
    return actions


def _evaluate_dataset(
    reader,
    trajectories,
    example_by_id,
    chunks_by_id,
    chunks_by_source,
    output_path: Path,
    manifest_config: Mapping,
    *,
    actions: tuple[str, ...],
    context_character_budget: int,
    context_allocation: str,
    context_source_order: str,
    context_compression: str,
    focused_sentences_per_source: int,
    log_every: int,
    overwrite: bool,
) -> list[dict]:
    manifest_path = output_path.with_suffix(".manifest.json")
    cached: dict[str, dict] = {}
    if output_path.exists() or manifest_path.exists():
        if overwrite:
            cached = {}
        elif _manifest_matches(manifest_path, manifest_config):
            cached = {str(row["id"]): row for row in _iter_jsonl(output_path)} if output_path.exists() else {}
        else:
            raise SystemExit(
                f"기존 STOP Answer 설정과 다릅니다: {output_path}\n"
                "다른 --output-root를 쓰거나 --overwrite를 지정하세요."
            )
    expected_ids = {str(row["id"]) for row in trajectories}
    rows_by_id = {row_id: row for row_id, row in cached.items() if row_id in expected_ids}
    if len(rows_by_id) == len(trajectories):
        _log(f"{manifest_config['dataset']}: Answer {len(rows_by_id):,}개 전체 재사용")
        return [rows_by_id[str(row["id"])] for row in trajectories]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite or not output_path.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    _write_manifest(manifest_path, manifest_config, "in_progress", len(rows_by_id), len(trajectories))
    started = perf_counter()
    newly_completed = 0
    with output_path.open(mode, encoding="utf-8") as destination:
        for trajectory in trajectories:
            row_id = str(trajectory["id"])
            if row_id in rows_by_id:
                continue
            example = example_by_id[row_id]
            action_results = {}
            restored_by_action = {
                action: _restore_sources(
                    trajectory["actions"][action],
                    chunks_by_id,
                    chunks_by_source,
                )
                for action in actions
            }
            initial_sources = restored_by_action.get(DIRECT_ACTION)
            if initial_sources is None:
                initial_sources = _restore_sources(
                    trajectory["actions"][DIRECT_ACTION],
                    chunks_by_id,
                    chunks_by_source,
                )
            initial_chunk_ids = tuple(
                chunk.chunk_id for chunk, _ in initial_sources
            )
            for action in actions:
                action_step = _action_step(action)
                sources = _order_sources_for_reader(
                    restored_by_action[action],
                    initial_chunk_ids,
                    context_source_order,
                )
                action_results[action] = _compact_action(
                    _score_sources(
                        reader,
                        example,
                        sources,
                        extra_candidates=action_step,
                        initial_retrieval_latency=0.0,
                        action_retrieval_latency=0.0,
                        context_character_budget=context_character_budget,
                        context_allocation=context_allocation,
                        context_compression=context_compression,
                        focus_queries=tuple(
                            str(item.get("query", ""))
                            for item in trajectory.get("planner_audit", [])[:action_step]
                            if str(item.get("query", "")).strip()
                        ),
                        focused_sentences_per_source=focused_sentences_per_source,
                    )
                )
            oracle_action = max(
                actions,
                key=lambda action: (
                    float(action_results[action]["f1"]),
                    float(action_results[action]["em"]),
                    -_action_step(action),
                ),
            )
            oracle = action_results[oracle_action]
            row = {
                "format_version": FORMAT_VERSION,
                "id": row_id,
                "dataset": example.dataset,
                "question": example.question,
                "hop_count": example.hop_count,
                "reasoning_type": example.reasoning_type,
                "actions": action_results,
                "oracle_action": oracle_action,
                "oracle_stop_step": _action_step(oracle_action),
                "oracle_em": float(oracle["em"]),
                "oracle_f1": float(oracle["f1"]),
                "stop_labels": _answer_stop_labels(action_results, actions),
                "planner_audit": trajectory.get("planner_audit", []),
            }
            direct = action_results.get(DIRECT_ACTION)
            if direct is not None:
                row.update({
                    "direct_em": float(direct["em"]),
                    "direct_f1": float(direct["f1"]),
                    "oracle_em_gain": float(oracle["em"]) - float(direct["em"]),
                    "oracle_f1_gain": float(oracle["f1"]) - float(direct["f1"]),
                })
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows_by_id[row_id] = row
            newly_completed += 1
            completed = len(rows_by_id)
            if completed % log_every == 0 or completed == len(trajectories):
                elapsed = perf_counter() - started
                eta = elapsed / max(1, newly_completed) * (len(trajectories) - completed)
                current = list(rows_by_id.values())
                cache_hits = sum(
                    int(result.get("reader_cache_hit", False))
                    for saved in current
                    for result in saved["actions"].values()
                )
                calls = len(current) * len(actions)
                if DIRECT_ACTION in actions:
                    score_summary = (
                        f"Direct/Oracle F1={_mean(current, 'direct_f1'):.3f}/"
                        f"{_mean(current, 'oracle_f1'):.3f}"
                    )
                else:
                    score_summary = "/".join(
                        f"{action} F1={_mean_action(current, action, 'f1'):.3f}"
                        for action in actions
                    )
                _log(
                    f"{example.dataset} Answer {completed:,}/{len(trajectories):,} "
                    f"({completed / len(trajectories):.1%}) | ETA {_duration(eta)} "
                    f"| {score_summary} | cache hit={cache_hits / max(1, calls):.0%}"
                )
                _write_manifest(manifest_path, manifest_config, "in_progress", completed, len(trajectories))
    _write_manifest(manifest_path, manifest_config, "ready", len(rows_by_id), len(trajectories))
    return [rows_by_id[str(row["id"])] for row in trajectories]


def _restore_sources(action: Mapping, chunks_by_id: Mapping, chunks_by_source: Mapping) -> list[tuple]:
    restored = []
    evidence = action.get("evidence", [])
    if evidence:
        for item in evidence:
            chunk = chunks_by_id.get(str(item.get("chunk_id")))
            if chunk is None:
                candidates = chunks_by_source.get(str(item.get("content_id")), [])
                chunk = candidates[0] if candidates else None
            if chunk is None:
                raise SystemExit(f"trajectory evidence chunk를 index에서 찾지 못했습니다: {item}")
            restored.append((chunk, float(item.get("score", 0.0))))
        return restored
    for content_id in action.get("evidence_content_ids", []):
        candidates = chunks_by_source.get(str(content_id), [])
        if not candidates:
            raise SystemExit(f"trajectory content를 index에서 찾지 못했습니다: {content_id}")
        restored.append((candidates[0], 0.0))
    return restored


def _order_sources_for_reader(
    sources: list[tuple],
    initial_chunk_ids: tuple[str, ...],
    order: str,
) -> list[tuple]:
    """Separate planner ordering from the reader's evidence presentation."""
    if order == "trajectory":
        return list(sources)
    if order != "initial_first":
        raise ValueError(f"지원하지 않는 context source order입니다: {order}")
    initial_position = {chunk_id: index for index, chunk_id in enumerate(initial_chunk_ids)}
    indexed = list(enumerate(sources))
    indexed.sort(
        key=lambda item: (
            0 if item[1][0].chunk_id in initial_position else 1,
            initial_position.get(item[1][0].chunk_id, item[0]),
            item[0],
        )
    )
    return [source for _, source in indexed]


def _answer_stop_labels(action_results: Mapping[str, Mapping], actions: tuple[str, ...]) -> list[dict]:
    labels = []
    for state_step in range(len(actions) - 1):
        current_action = actions[state_step]
        future = actions[state_step + 1 :]
        best_future = max(
            future,
            key=lambda action: (
                float(action_results[action]["f1"]),
                float(action_results[action]["em"]),
                -_action_step(action),
            ),
        )
        current = action_results[current_action]
        future_result = action_results[best_future]
        continue_search = (
            float(future_result["f1"]),
            float(future_result["em"]),
        ) > (
            float(current["f1"]),
            float(current["em"]),
        )
        labels.append(
            {
                "state_step": state_step,
                "label": "CONTINUE" if continue_search else "STOP",
                "current_action": current_action,
                "best_future_action": best_future,
                "f1_gain": float(future_result["f1"]) - float(current["f1"]),
                "em_gain": float(future_result["em"]) - float(current["em"]),
            }
        )
    return labels


def _compact_action(result: Mapping) -> dict:
    keep = (
        "prediction",
        "matched_gold",
        "em",
        "f1",
        "source_recall",
        "all_support_recall",
        "reader_latency_seconds",
        "reader_cache_hit",
        "context_hash",
        "context_characters",
    )
    return {
        **{key: result[key] for key in keep if key in result},
        "evidence_content_ids": [str(item["content_id"]) for item in result.get("evidence", ())],
    }


def _action_step(action: str) -> int:
    if action == DIRECT_ACTION:
        return 0
    return int(action.rsplit("_", 1)[1])


def _trajectory_signature(manifest: Mapping) -> str:
    ignored = {"status", "completed_rows", "requested_rows", "queries_by_hop", "queries_per_hop"}
    payload = {key: value for key, value in manifest.items() if key not in ignored}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report(rows: list[dict], name: str, actions: tuple[str, ...]) -> None:
    values = "/".join(f"{_mean_action(rows, action, 'f1'):.3f}" for action in actions)
    if DIRECT_ACTION not in actions:
        _log(f"{name}: n={len(rows):,} | action F1 {'/'.join(actions)}={values}")
        return
    direct = _mean(rows, "direct_f1")
    oracle = _mean(rows, "oracle_f1")
    distribution = dict(sorted(Counter(str(row["oracle_action"]) for row in rows).items()))
    _log(
        f"{name}: n={len(rows):,} | action F1 {'/'.join(actions)}={values} "
        f"| Direct/Oracle={direct:.3f}/{oracle:.3f} | gain={oracle - direct:+.3f} "
        f"| Oracle STOP={distribution}"
    )
    for hop in sorted({row.get("hop_count") for row in rows if row.get("hop_count") is not None}):
        selected = [row for row in rows if row.get("hop_count") == hop]
        _log(
            f"  hop={hop}: n={len(selected):,} | Direct/Oracle F1="
            f"{_mean(selected, 'direct_f1'):.3f}/{_mean(selected, 'oracle_f1'):.3f} "
            f"| gain={_mean(selected, 'oracle_f1') - _mean(selected, 'direct_f1'):+.3f}"
        )


def _report_macro(rows: list[dict], datasets, actions: tuple[str, ...]) -> None:
    groups = [[row for row in rows if row["dataset"] == dataset] for dataset in datasets]
    groups = [group for group in groups if group]
    action_values = "/".join(
        f"{float(np.mean([_mean_action(group, action, 'f1') for group in groups])):.3f}"
        for action in actions
    )
    if DIRECT_ACTION not in actions:
        _log(f"Macro: action F1 {'/'.join(actions)}={action_values}")
        return
    direct = float(np.mean([_mean(group, "direct_f1") for group in groups]))
    oracle = float(np.mean([_mean(group, "oracle_f1") for group in groups]))
    _log(
        f"Macro: action F1 {'/'.join(actions)}={action_values} | Direct/Oracle="
        f"{direct:.3f}/{oracle:.3f} | gain={oracle - direct:+.3f}"
    )


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _mean_action(rows: list[dict], action: str, key: str) -> float:
    return float(np.mean([float(row["actions"][action][key]) for row in rows])) if rows else 0.0


def _manifest_matches(path: Path, expected: Mapping) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        ignored = {"status", "completed_rows", "requested_rows"}
        return all(saved.get(key) == value for key, value in expected.items() if key not in ignored)
    except (OSError, ValueError, TypeError):
        return False


def _write_manifest(path: Path, config: Mapping, status: str, completed: int, requested: int) -> None:
    payload = dict(config)
    payload.update({"status": status, "completed_rows": completed, "requested_rows": requested})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
