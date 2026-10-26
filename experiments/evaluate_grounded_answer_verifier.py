"""Evaluate one uniform conservative draft verifier on every QA dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.evaluate_answer_oracle import _load_fixed_retriever
from experiments.evaluate_stop_answer_oracle import _restore_sources
from experiments.train_multihop_policy import (
    SUPPORTED_DATASETS,
    _corpus_fingerprint,
    _dataset_seed,
    _duration,
    _iter_jsonl,
    load_processed_examples,
)
from src.answer_verifier import (
    ADJUDICATOR_PROMPT_VERSION,
    PROMPT_VERSION,
    FrozenCorrectionAdjudicator,
    FrozenGroundedAnswerVerifier,
)
from src.qa_evaluation import AnswerCache, ReaderConfig, answer_score, evidence_context, ollama_model_identity
from src.settings import settings


ACTION = "planned_continue_3"
FORMAT_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="동일 grounded draft verifier 세 데이터셋 평가")
    parser.add_argument("--trajectory-root", default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000")
    parser.add_argument("--baseline-answer-root", default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_validation_2000")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-path", default="data/benchmarks/multihop/answer_verifier_cache.sqlite3")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-rows-per-dataset", type=int, default=100)
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--verifier-model", default=settings.ollama_model)
    parser.add_argument("--verifier-max-tokens", type=int, default=256)
    parser.add_argument("--verifier-timeout", type=int, default=300)
    parser.add_argument("--verifier-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(args.corpus_examples_per_dataset, args.max_rows_per_dataset, args.context_character_budget) < 1:
        parser.error("corpus, rows, context는 1 이상이어야 합니다.")
    return args


def _manifest_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    return all(saved.get(key) == value for key, value in expected.items())


def _write_manifest(path: Path, config: dict, status: str, completed: int, requested: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**config, "status": status, "completed_rows": completed, "requested_rows": requested}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_dataset(
    args,
    dataset: str,
    verifier,
    adjudicator,
    verifier_config: ReaderConfig,
    shared_embedding_model,
    shared_reranker,
):
    trajectories = list(_iter_jsonl(Path(args.trajectory_root) / f"{dataset}_{args.split}.jsonl"))[: args.max_rows_per_dataset]
    baseline_rows = list(_iter_jsonl(Path(args.baseline_answer_root) / f"{dataset}_{args.split}.jsonl"))
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    corpus = load_processed_examples(
        Path(args.processed_root) / dataset / f"{args.split}.jsonl",
        args.corpus_examples_per_dataset,
        _dataset_seed(args.corpus_seed, dataset),
    )
    example_by_id = {row.policy_id: row for row in corpus}
    fingerprint = _corpus_fingerprint(corpus)
    retriever = _load_fixed_retriever(
        Path(args.index_root) / args.split / dataset,
        fingerprint,
        embedding_model=shared_embedding_model,
        reranker=shared_reranker,
    )
    chunks_by_id = {chunk.chunk_id: chunk for chunk in retriever.chunks}
    chunks_by_source: dict[str, list] = {}
    for chunk in retriever.chunks:
        chunks_by_source.setdefault(chunk.source, []).append(chunk)

    output = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
    manifest_path = output.with_suffix(".manifest.json")
    config = {
        "format_version": FORMAT_VERSION,
        "experiment": "uniform_grounded_draft_verifier_with_independent_adjudication",
        "dataset": dataset,
        "split": args.split,
        "action": ACTION,
        "corpus_fingerprint": fingerprint,
        "context_character_budget": args.context_character_budget,
        "verifier": {**asdict(verifier_config), "prompt_version": PROMPT_VERSION},
        "adjudicator_prompt_version": ADJUDICATOR_PROMPT_VERSION,
    }
    cached: dict[str, dict] = {}
    if output.exists() or manifest_path.exists():
        if args.overwrite:
            cached = {}
        elif _manifest_matches(manifest_path, config):
            cached = {str(row["id"]): row for row in _iter_jsonl(output)} if output.exists() else {}
        else:
            raise SystemExit(f"기존 verifier 설정과 다릅니다: {output}")
    expected_ids = {str(row["id"]) for row in trajectories}
    rows_by_id = {row_id: row for row_id, row in cached.items() if row_id in expected_ids}
    if len(rows_by_id) == len(trajectories):
        return [rows_by_id[str(row["id"])] for row in trajectories], retriever.model, retriever.reranker

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    _write_manifest(manifest_path, config, "in_progress", len(rows_by_id), len(trajectories))
    started = perf_counter()
    newly_completed = 0
    with output.open(mode, encoding="utf-8") as destination:
        for trajectory in trajectories:
            row_id = str(trajectory["id"])
            if row_id in rows_by_id:
                continue
            example = example_by_id[row_id]
            baseline = baseline_by_id[row_id]["actions"][ACTION]
            draft = str(baseline["prediction"])
            sources = _restore_sources(trajectory["actions"][ACTION], chunks_by_id, chunks_by_source)
            context, _ = evidence_context(sources, args.context_character_budget)
            result = verifier.verify(example.question, context, draft)
            adjudication = None
            if result.decision == "correct" and result.prediction != draft:
                adjudication = adjudicator.adjudicate(
                    example.question,
                    context,
                    draft,
                    result.prediction,
                )
            accepted = bool(adjudication and adjudication.accept)
            final_prediction = result.prediction if accepted else draft
            score = answer_score(
                final_prediction,
                str(example.answer),
                example.answer_aliases,
                dataset=dataset,
            )
            row = {
                "format_version": FORMAT_VERSION,
                "id": row_id,
                "dataset": dataset,
                "question": example.question,
                "hop_count": example.hop_count,
                "reasoning_type": example.reasoning_type,
                "draft_prediction": draft,
                "draft_em": float(baseline["em"]),
                "draft_f1": float(baseline["f1"]),
                "verifier_decision": result.decision,
                "proposed_prediction": result.prediction,
                "adjudicator_decision": (
                    "accept" if accepted else "reject" if adjudication is not None else "not_needed"
                ),
                "prediction": final_prediction,
                "em": score.em,
                "f1": score.f1,
                "changed": final_prediction != draft,
                "verification_chain": list(result.verification_chain),
                "independent_chain": list(adjudication.independent_chain) if adjudication else [],
                "cache_hit": result.cache_hit,
                "latency_seconds": result.latency_seconds,
            }
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows_by_id[row_id] = row
            newly_completed += 1
            completed = len(rows_by_id)
            if completed % args.log_every == 0 or completed == len(trajectories):
                current = list(rows_by_id.values())
                elapsed = perf_counter() - started
                eta = elapsed / max(1, newly_completed) * (len(trajectories) - completed)
                print(
                    f"{dataset} verifier {completed:,}/{len(trajectories):,} | ETA {_duration(eta)} | "
                    f"draft/final F1={np.mean([r['draft_f1'] for r in current]):.3f}/"
                    f"{np.mean([r['f1'] for r in current]):.3f} | "
                    f"correct={np.mean([r['changed'] for r in current]):.1%}",
                    flush=True,
                )
                _write_manifest(manifest_path, config, "in_progress", completed, len(trajectories))
    _write_manifest(manifest_path, config, "ready", len(rows_by_id), len(trajectories))
    return [rows_by_id[str(row["id"])] for row in trajectories], retriever.model, retriever.reranker


def main() -> None:
    args = _arguments()
    session = requests.Session()
    model, digest = ollama_model_identity(args.ollama_host, args.verifier_model, session=session)
    config = ReaderConfig(
        host=args.ollama_host,
        model=model,
        model_digest=digest,
        max_tokens=args.verifier_max_tokens,
        timeout_seconds=args.verifier_timeout,
        seed=args.verifier_seed,
        temperature=0.0,
        keep_alive=args.ollama_keep_alive,
        prompt_version=PROMPT_VERSION,
        thinking=False,
    )
    all_rows = []
    shared_model = shared_reranker = None
    with AnswerCache(Path(args.cache_path)) as cache:
        verifier = FrozenGroundedAnswerVerifier(config, cache, session=session)
        adjudicator = FrozenCorrectionAdjudicator(config, cache, session=session)
        for dataset in args.datasets:
            rows, shared_model, shared_reranker = _run_dataset(
                args, dataset, verifier, adjudicator, config, shared_model, shared_reranker
            )
            all_rows.extend(rows)
            print(
                f"{dataset}: n={len(rows):,} EM/F1="
                f"{np.mean([r['em'] for r in rows]):.3f}/{np.mean([r['f1'] for r in rows]):.3f} "
                f"draft={np.mean([r['draft_em'] for r in rows]):.3f}/"
                f"{np.mean([r['draft_f1'] for r in rows]):.3f} decisions="
                f"{dict(Counter(r['adjudicator_decision'] for r in rows))}",
                flush=True,
            )
    session.close()


if __name__ == "__main__":
    main()
