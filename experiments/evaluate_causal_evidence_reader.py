"""Evaluate a learned compact evidence bottleneck with the frozen QA reader."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import requests

from experiments.evaluate_answer_oracle import _load_fixed_retriever
from experiments.train_counterfactual_proof_policy import _candidates, _load_split
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _corpus_fingerprint, _duration, _iter_jsonl
from src.causal_evidence import (
    evidence_width_features,
    proof_policy_features,
    select_topology_aware_indices,
)
from src.qa_evaluation import (
    PROMPT_VERSION,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    answer_score,
    evidence_context,
    ollama_model_identity,
)
from src.settings import settings


ACTION = "planned_continue_3"
FORMAT_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual evidence bottleneck reader pilot")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument(
        "--trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument(
        "--baseline-answer-root",
        default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_validation_2000",
    )
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--policy-model", default="data/models/counterfactual_proof_completion_policy.joblib")
    parser.add_argument("--width-model")
    parser.add_argument("--topology-aware", action="store_true")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-path", default="data/benchmarks/multihop/answer_cache.sqlite3")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-rows-per-dataset", type=int, default=100)
    parser.add_argument("--selected-documents", type=int, default=4)
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--reader-model", default=settings.ollama_model)
    parser.add_argument("--reader-max-tokens", type=int, default=64)
    parser.add_argument("--reader-timeout", type=int, default=300)
    parser.add_argument("--reader-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(args.max_rows_per_dataset, args.selected_documents, args.context_character_budget) < 1:
        parser.error("rows/documents/context는 1 이상이어야 합니다.")
    return args


def _manifest_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    return all(saved.get(key) == value for key, value in expected.items())


def _write_manifest(path: Path, config: dict, status: str, completed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**config, "status": status, "completed_rows": completed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_dataset(
    args,
    dataset,
    policy,
    width_bundle,
    reader,
    reader_config,
    shared_model,
    shared_reranker,
):
    joined, documents = _load_split(
        args,
        dataset,
        args.split,
        Path(args.trajectory_root),
        args.max_rows_per_dataset,
    )
    baseline_rows = list(
        _iter_jsonl(Path(args.baseline_answer_root) / f"{dataset}_{args.split}.jsonl")
    )
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    corpus = [example for _, example in joined]
    # The fixed index fingerprint covers the full 2,000-example corpus, not
    # only the reader pilot cohort, so load it separately through _load_split.
    full_joined, full_documents = _load_split(
        args,
        dataset,
        args.split,
        Path(args.trajectory_root),
        2000,
    )
    del full_documents
    full_examples = list({example.policy_id: example for _, example in full_joined}.values())
    fingerprint = _corpus_fingerprint(full_examples)
    retriever = _load_fixed_retriever(
        Path(args.index_root) / args.split / dataset,
        fingerprint,
        embedding_model=shared_model,
        reranker=shared_reranker,
    )
    chunks = {chunk.chunk_id: chunk for chunk in retriever.chunks}

    output = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
    manifest_path = output.with_suffix(".manifest.json")
    config = {
        "format_version": FORMAT_VERSION,
        "experiment": "counterfactual_evidence_bottleneck_reader",
        "dataset": dataset,
        "split": args.split,
        "requested_rows": len(joined),
        "selected_documents": args.selected_documents,
        "width_model": str(Path(args.width_model)) if args.width_model else None,
        "topology_aware": args.topology_aware,
        "context_character_budget": args.context_character_budget,
        "policy_model": str(Path(args.policy_model)),
        "reader": asdict(reader_config),
    }
    cached = {}
    if output.exists() or manifest_path.exists():
        if args.overwrite:
            cached = {}
        elif _manifest_matches(manifest_path, config):
            cached = {str(row["id"]): row for row in _iter_jsonl(output)} if output.exists() else {}
        else:
            raise SystemExit(f"기존 evidence-reader 설정과 다릅니다: {output}")
    expected_ids = {example.policy_id for _, example in joined}
    rows_by_id = {row_id: row for row_id, row in cached.items() if row_id in expected_ids}
    if len(rows_by_id) == len(joined):
        return [rows_by_id[example.policy_id] for _, example in joined], retriever.model, retriever.reranker

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    _write_manifest(manifest_path, config, "in_progress", len(rows_by_id))
    started = perf_counter()
    new = 0
    with output.open(mode, encoding="utf-8") as destination:
        for trajectory, example in joined:
            row_id = example.policy_id
            if row_id in rows_by_id:
                continue
            candidates = _candidates(trajectory)
            matrix = proof_policy_features(example.question, candidates, documents, ())
            probabilities = policy.predict_proba(matrix)[:, 1][:-1]
            if width_bundle is not None:
                width_features = evidence_width_features(
                    policy, example.question, candidates, documents
                )
                raw_width = float(width_bundle["model"].predict(width_features.reshape(1, -1))[0])
                selected_width = int(
                    np.clip(
                        np.ceil(raw_width),
                        int(width_bundle["minimum_width"]),
                        min(int(width_bundle["maximum_width"]), len(candidates)),
                    )
                )
            else:
                selected_width = min(args.selected_documents, len(candidates))
            selected_indices = (
                np.asarray(
                    select_topology_aware_indices(
                        example.question,
                        candidates,
                        documents,
                        probabilities,
                        selected_width,
                    ),
                    dtype="int64",
                )
                if args.topology_aware
                else np.argsort(-probabilities)[:selected_width]
            )
            sources = [
                (chunks[str(candidates[index]["chunk_id"])], float(probabilities[index]))
                for index in selected_indices
                if str(candidates[index]["chunk_id"]) in chunks
            ]
            context, provenance = evidence_context(sources, args.context_character_budget)
            result = reader.answer(example.question, context)
            score = answer_score(
                result.prediction,
                str(example.answer),
                example.answer_aliases,
                dataset=dataset,
            )
            baseline = baseline_by_id[row_id]["actions"][ACTION]
            row = {
                "format_version": FORMAT_VERSION,
                "id": row_id,
                "dataset": dataset,
                "question": example.question,
                "reasoning_type": example.reasoning_type,
                "prediction": result.prediction,
                "em": score.em,
                "f1": score.f1,
                "baseline_prediction": baseline["prediction"],
                "baseline_em": float(baseline["em"]),
                "baseline_f1": float(baseline["f1"]),
                "selected_indices": [int(index) for index in selected_indices],
                "selected_width": selected_width,
                "selected_probabilities": [float(probabilities[index]) for index in selected_indices],
                "evidence": list(provenance),
                "cache_hit": result.cache_hit,
                "latency_seconds": result.latency_seconds,
            }
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows_by_id[row_id] = row
            new += 1
            completed = len(rows_by_id)
            if completed % args.log_every == 0 or completed == len(joined):
                current = list(rows_by_id.values())
                elapsed = perf_counter() - started
                eta = elapsed / max(1, new) * (len(joined) - completed)
                print(
                    f"{dataset} compact {completed}/{len(joined)} | ETA {_duration(eta)} | "
                    f"baseline/compact F1={np.mean([r['baseline_f1'] for r in current])*100:.2f}/"
                    f"{np.mean([r['f1'] for r in current])*100:.2f}",
                    flush=True,
                )
                _write_manifest(manifest_path, config, "in_progress", completed)
    _write_manifest(manifest_path, config, "ready", len(rows_by_id))
    return [rows_by_id[example.policy_id] for _, example in joined], retriever.model, retriever.reranker


def main() -> None:
    args = _arguments()
    bundle = joblib.load(args.policy_model)
    policy = bundle["model"]
    width_bundle = joblib.load(args.width_model) if args.width_model else None
    session = requests.Session()
    model, digest = ollama_model_identity(args.ollama_host, args.reader_model, session=session)
    reader_config = ReaderConfig(
        host=args.ollama_host,
        model=model,
        model_digest=digest,
        max_tokens=args.reader_max_tokens,
        timeout_seconds=args.reader_timeout,
        seed=args.reader_seed,
        temperature=0.0,
        keep_alive=args.ollama_keep_alive,
        prompt_version=PROMPT_VERSION,
        thinking=False,
    )
    shared_model = shared_reranker = None
    with AnswerCache(Path(args.cache_path)) as cache:
        reader = FrozenOllamaReader(reader_config, cache, session=session)
        for dataset in args.datasets:
            rows, shared_model, shared_reranker = _run_dataset(
                args,
                dataset,
                policy,
                width_bundle,
                reader,
                reader_config,
                shared_model,
                shared_reranker,
            )
            print(
                f"{dataset}: n={len(rows)} baseline EM/F1="
                f"{np.mean([r['baseline_em'] for r in rows])*100:.2f}/"
                f"{np.mean([r['baseline_f1'] for r in rows])*100:.2f} compact EM/F1="
                f"{np.mean([r['em'] for r in rows])*100:.2f}/"
                f"{np.mean([r['f1'] for r in rows])*100:.2f}",
                flush=True,
            )
    session.close()


if __name__ == "__main__":
    main()
