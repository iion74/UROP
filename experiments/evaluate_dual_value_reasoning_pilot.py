"""Evaluate adaptive reasoning compute on the dual-value router's REASON queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.evaluate_context_answer_oracle import _iter_processed_examples
from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from experiments.train_reader_free_student_policy import DECISION_STEP
from src.document_loader import SourceDocument, chunk_documents
from src.qa_evaluation import (
    EVIDENCE_CHAIN_PROMPT_VERSION,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    answer_score,
    evidence_context,
    ollama_model_identity,
)
from src.settings import settings


FORMAT_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-value adaptive-thinking pilot")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument(
        "--prefix-root",
        default="data/benchmarks/multihop/context_reader_aware/full_validation",
    )
    parser.add_argument(
        "--reason-queue",
        default=(
            "data/benchmarks/multihop/context_reader_aware/analysis/"
            "dual_value_router_multiseed.reason_queue.jsonl"
        ),
    )
    parser.add_argument(
        "--all-questions",
        action="store_true",
        help="표본의 모든 prefix step2 답변에 thinking/judge의 반사실 결과를 생성합니다.",
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--rows-per-dataset", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        default="data/benchmarks/multihop/context_reader_aware/dual_value_reasoning_pilot",
    )
    parser.add_argument(
        "--exclude-root",
        action="append",
        default=[],
        help="선행 pilot의 *_validation.jsonl ID를 제외합니다. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--answer-cache",
        default="data/benchmarks/multihop/dual_value_reasoning_answer_cache.sqlite3",
    )
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--reader-max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default="0")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.rows_per_dataset < 0 or min(
        args.context_character_budget,
        args.reader_max_tokens,
        args.timeout,
        args.log_every,
    ) < 1:
        parser.error("rows/context/token/timeout/log는 1 이상이어야 합니다.")
    return args


def _stable_sample(rows: list[dict], maximum: int, seed: int, dataset: str) -> list[dict]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{dataset}\0{row['id']}".encode("utf-8")
        ).digest(),
    )
    return ranked[:maximum]


def _context(example, prefix_row: dict, character_budget: int) -> tuple[str, tuple[dict, ...]]:
    documents = [
        SourceDocument(source=document.content_id, title=document.title, text=document.text)
        for document in example.documents
    ]
    chunks = {
        chunk.chunk_id: chunk
        for chunk in chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    }
    action = prefix_row["actions"][ACTIONS[DECISION_STEP]]
    sources = []
    for item in action.get("candidate_evidence", ()):
        chunk_id = str(item["chunk_id"])
        if chunk_id not in chunks:
            raise ValueError(f"{prefix_row['id']}: chunk를 복원할 수 없습니다: {chunk_id}")
        sources.append((chunks[chunk_id], float(item["score"])))
    context, provenance = evidence_context(sources, character_budget)
    expected = [str(item["content_id"]) for item in action.get("evidence", ())]
    actual = [str(item["content_id"]) for item in provenance]
    if actual != expected:
        raise ValueError(f"{prefix_row['id']}: 재구성한 reader context가 원본과 다릅니다.")
    return context, provenance


def _manifest_matches(path: Path, config: dict) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return all(saved.get(key) == value for key, value in config.items())


def _write_manifest(path: Path, config: dict, status: str, completed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {**config, "status": status, "completed_rows": completed},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _summary(rows: list[dict]) -> dict:
    baseline_em = float(np.mean([row["baseline_em"] for row in rows]))
    baseline_f1 = float(np.mean([row["baseline_f1"] for row in rows]))
    repair_em = float(np.mean([row["repair_em"] for row in rows]))
    repair_f1 = float(np.mean([row["repair_f1"] for row in rows]))
    return {
        "count": len(rows),
        "baseline_em": baseline_em,
        "baseline_f1": baseline_f1,
        "repair_em": repair_em,
        "repair_f1": repair_f1,
        "em_change_points": (repair_em - baseline_em) * 100.0,
        "f1_change_points": (repair_f1 - baseline_f1) * 100.0,
        "wrong_to_exact": sum(row["baseline_em"] < 1.0 and row["repair_em"] == 1.0 for row in rows),
        "exact_to_wrong": sum(row["baseline_em"] == 1.0 and row["repair_em"] < 1.0 for row in rows),
        "average_latency_seconds": float(np.mean([row["repair_latency_seconds"] for row in rows])),
        "average_prompt_tokens": float(
            np.mean([row["repair_prompt_tokens"] or 0 for row in rows])
        ),
        "average_generated_tokens": float(
            np.mean([row["repair_generated_tokens"] or 0 for row in rows])
        ),
    }


def _run_dataset(args, dataset: str, selected: list[dict], reader, reader_config) -> dict:
    prefix_by_id = {
        str(row["id"]): row
        for row in _iter_jsonl(Path(args.prefix_root) / f"{dataset}_{args.split}.jsonl")
    }
    selected_ids = {str(row["id"]) for row in selected}
    examples_by_id = {
        example.policy_id: example
        for example in _iter_processed_examples(
            Path(args.processed_root) / dataset / f"{args.split}.jsonl"
        )
        if example.policy_id in selected_ids
    }
    output = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
    manifest = output.with_suffix(".manifest.json")
    selection_digest = hashlib.sha256(
        "\n".join(sorted(str(row["id"]) for row in selected)).encode("utf-8")
    ).hexdigest()
    config = {
        "format_version": FORMAT_VERSION,
        "experiment": "dual_value_adaptive_thinking_pilot",
        "dataset": dataset,
        "sample_seed": args.sample_seed,
        "requested_rows": len(selected),
        "selection_ids_sha256": selection_digest,
        "context_character_budget": args.context_character_budget,
        "reader": asdict(reader_config),
    }
    cached = {}
    if output.exists() or manifest.exists():
        if args.overwrite:
            cached = {}
        elif _manifest_matches(manifest, config):
            cached = {str(row["id"]): row for row in _iter_jsonl(output)} if output.exists() else {}
        else:
            raise SystemExit(f"기존 adaptive-thinking 설정과 다릅니다: {output}")
    if len(cached) == len(selected):
        return _summary([cached[str(row["id"])] for row in selected])
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output.exists() else "a"
    if mode == "w":
        cached = {}
    _write_manifest(manifest, config, "in_progress", len(cached))
    started = perf_counter()
    new = 0
    with output.open(mode, encoding="utf-8") as destination:
        for queue_row in selected:
            identifier = str(queue_row["id"])
            if identifier in cached:
                continue
            example = examples_by_id[identifier]
            prefix = prefix_by_id[identifier]
            action = prefix["actions"][ACTIONS[DECISION_STEP]]
            context, provenance = _context(example, prefix, args.context_character_budget)
            answer = reader.answer(example.question, context)
            score = answer_score(
                answer.prediction,
                str(example.answer),
                example.answer_aliases,
                dataset=dataset,
            )
            record = {
                "format_version": FORMAT_VERSION,
                "id": identifier,
                "dataset": dataset,
                "question": example.question,
                "route_votes": queue_row["route_votes"],
                "baseline_prediction": str(action["prediction"]),
                "baseline_em": float(action["em"]),
                "baseline_f1": float(action["f1"]),
                "repair_prediction": answer.prediction,
                "repair_em": score.em,
                "repair_f1": score.f1,
                "repair_cache_hit": answer.cache_hit,
                "repair_latency_seconds": answer.latency_seconds,
                "repair_prompt_tokens": answer.prompt_eval_count,
                "repair_generated_tokens": answer.eval_count,
                "evidence": list(provenance),
            }
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            cached[identifier] = record
            new += 1
            completed = len(cached)
            if completed % args.log_every == 0 or completed == len(selected):
                current = list(cached.values())
                elapsed = perf_counter() - started
                eta = elapsed / max(1, new) * max(0, len(selected) - completed)
                summary = _summary(current)
                print(
                    f"{dataset} thinking {completed}/{len(selected)} | ETA {_duration(eta)} | "
                    f"EM={summary['baseline_em']*100:.2f}->{summary['repair_em']*100:.2f} "
                    f"F1={summary['baseline_f1']*100:.2f}->{summary['repair_f1']*100:.2f}",
                    flush=True,
                )
                _write_manifest(manifest, config, "in_progress", completed)
    rows = [cached[str(row["id"])] for row in selected]
    _write_manifest(manifest, config, "ready", len(rows))
    return _summary(rows)


def main() -> None:
    args = _arguments()
    if args.all_questions:
        queue = []
        for dataset in args.datasets:
            for prefix in _iter_jsonl(Path(args.prefix_root) / f"{dataset}_{args.split}.jsonl"):
                action = prefix["actions"][ACTIONS[DECISION_STEP]]
                queue.append(
                    {
                        "id": str(prefix["id"]),
                        "dataset": dataset,
                        "question": str(prefix["question"]),
                        "route_votes": {},
                        "baseline_prediction": str(action["prediction"]),
                        "baseline_em": float(action["em"]),
                        "baseline_f1": float(action["f1"]),
                    }
                )
    else:
        queue = list(_iter_jsonl(Path(args.reason_queue)))
    excluded = {dataset: set() for dataset in args.datasets}
    for root in args.exclude_root:
        for dataset in args.datasets:
            path = Path(root) / f"{dataset}_{args.split}.jsonl"
            if not path.exists():
                raise SystemExit(f"제외할 선행 pilot가 없습니다: {path}")
            excluded[dataset].update(str(row["id"]) for row in _iter_jsonl(path))
    selected = {
        dataset: (
            sorted(
                [
                    row
                    for row in queue
                    if row["dataset"] == dataset and str(row["id"]) not in excluded[dataset]
                ],
                key=lambda row: str(row["id"]),
            )
            if args.rows_per_dataset == 0
            else _stable_sample(
                [
                    row
                    for row in queue
                    if row["dataset"] == dataset and str(row["id"]) not in excluded[dataset]
                ],
                args.rows_per_dataset,
                args.sample_seed,
                dataset,
            )
        )
        for dataset in args.datasets
    }
    if any(len(rows) < args.rows_per_dataset for rows in selected.values()):
        raise SystemExit("REASON queue가 요청한 표본 수보다 작습니다.")
    session = requests.Session()
    model, digest = ollama_model_identity(args.ollama_host, args.model, session=session)
    reader_config = ReaderConfig(
        host=args.ollama_host,
        model=model,
        model_digest=digest,
        max_tokens=args.reader_max_tokens,
        timeout_seconds=args.timeout,
        seed=args.model_seed,
        temperature=0.0,
        keep_alive=args.ollama_keep_alive,
        prompt_version=EVIDENCE_CHAIN_PROMPT_VERSION,
        thinking=True,
    )
    summaries = {}
    with ExitStack() as stack:
        cache = stack.enter_context(AnswerCache(Path(args.answer_cache)))
        reader = FrozenOllamaReader(reader_config, cache, session=session)
        for dataset in args.datasets:
            summaries[dataset] = _run_dataset(
                args, dataset, selected[dataset], reader, reader_config
            )
            result = summaries[dataset]
            print(
                f"{dataset}: n={result['count']} EM={result['baseline_em']*100:.2f}->"
                f"{result['repair_em']*100:.2f} ({result['em_change_points']:+.2f}) "
                f"F1={result['baseline_f1']*100:.2f}->{result['repair_f1']*100:.2f} "
                f"({result['f1_change_points']:+.2f})",
                flush=True,
            )
    session.close()
    summary_path = Path(args.output_root) / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "experiment": "dual_value_adaptive_thinking_pilot",
                "sample_seed": args.sample_seed,
                "rows_per_dataset": args.rows_per_dataset,
                "reader": asdict(reader_config),
                "datasets": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
