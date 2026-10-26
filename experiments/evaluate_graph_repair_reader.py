"""Evaluate local graph proof repair against the frozen topology reader."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import requests

from experiments.evaluate_answer_oracle import _load_fixed_retriever
from experiments.train_counterfactual_proof_policy import _candidates, _load_split
from experiments.train_graph_repair_policy import _base_selection
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _corpus_fingerprint, _duration, _iter_jsonl
from src.proof_repair import (
    build_graph_repair_proposals,
    graph_repair_features,
    repair_syndrome_features,
    select_graph_repairs,
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


FORMAT_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local graph proof repair reader evaluation")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument(
        "--trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument(
        "--baseline-root",
        default="data/benchmarks/multihop/causal_evidence_reader/topology3_adaptive_validation_2000",
    )
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--support-model", default="data/models/counterfactual_proof_completion_policy.joblib")
    parser.add_argument("--width-model", default="data/models/evidence_width_policy.joblib")
    parser.add_argument("--repair-model", default="data/models/local_graph_proof_repair_policy.joblib")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-path", default="data/benchmarks/multihop/answer_cache.sqlite3")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-rows-per-dataset", type=int, default=100)
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
    if min(args.max_rows_per_dataset, args.context_character_budget, args.log_every) < 1:
        parser.error("rows/context/log 값은 1 이상이어야 합니다.")
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
    support_model,
    width_bundle,
    repair_bundle,
    reader,
    reader_config,
    shared_model,
    shared_reranker,
):
    joined, documents = _load_split(
        args, dataset, args.split, Path(args.trajectory_root), args.max_rows_per_dataset
    )
    baseline_rows = list(_iter_jsonl(Path(args.baseline_root) / f"{dataset}_{args.split}.jsonl"))
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    full_joined, _ = _load_split(args, dataset, args.split, Path(args.trajectory_root), 2000)
    full_examples = list({example.policy_id: example for _, example in full_joined}.values())
    retriever = _load_fixed_retriever(
        Path(args.index_root) / args.split / dataset,
        _corpus_fingerprint(full_examples),
        embedding_model=shared_model,
        reranker=shared_reranker,
    )
    chunks = {chunk.chunk_id: chunk for chunk in retriever.chunks}
    output = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
    manifest = output.with_suffix(".manifest.json")
    config = {
        "format_version": FORMAT_VERSION,
        "experiment": "local_graph_proof_repair_reader",
        "dataset": dataset,
        "split": args.split,
        "requested_rows": len(joined),
        "baseline_root": str(Path(args.baseline_root)),
        "support_model": str(Path(args.support_model)),
        "width_model": str(Path(args.width_model)),
        "repair_model": str(Path(args.repair_model)),
        "repair_threshold": float(
            repair_bundle.get("syndrome_threshold", repair_bundle.get("threshold", 1.0))
        ),
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
            raise SystemExit(f"기존 graph-repair reader 설정과 다릅니다: {output}")
    expected_ids = {example.policy_id for _, example in joined}
    rows_by_id = {key: row for key, row in cached.items() if key in expected_ids}
    if len(rows_by_id) == len(joined):
        return [rows_by_id[example.policy_id] for _, example in joined], retriever.model, retriever.reranker
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    _write_manifest(manifest, config, "in_progress", len(rows_by_id))
    started = __import__("time").perf_counter()
    new = 0
    with output.open(mode, encoding="utf-8") as destination:
        for trajectory, example in joined:
            if example.policy_id in rows_by_id:
                continue
            candidates = _candidates(trajectory)
            selected, support_probabilities = _base_selection(
                support_model, width_bundle, example.question, candidates, documents
            )
            proposals = build_graph_repair_proposals(
                retriever,
                candidates,
                selected,
                neighbors_per_seed=int(repair_bundle["neighbors_per_seed"]),
                maximum_proposals=int(repair_bundle["maximum_proposals"]),
            )
            repair_features = graph_repair_features(
                example.question,
                candidates,
                selected,
                support_probabilities,
                proposals,
                documents,
            )
            if repair_bundle.get("method") == "factorized_local_graph_proof_repair":
                repair_probabilities = (
                    repair_bundle["ranker"].predict_proba(repair_features)[:, 1]
                    if len(repair_features)
                    else np.empty(0, dtype="float64")
                )
                syndrome_features = repair_syndrome_features(
                    repair_features, repair_probabilities, len(selected)
                )
                syndrome_probability = float(
                    repair_bundle["syndrome"].predict_proba(syndrome_features.reshape(1, -1))[0, 1]
                )
                repairs = (
                    (int(np.argmax(repair_probabilities)),)
                    if len(repair_probabilities)
                    and syndrome_probability >= float(repair_bundle["syndrome_threshold"])
                    else ()
                )
            else:
                repair_probabilities = (
                    repair_bundle["model"].predict_proba(repair_features)[:, 1]
                    if len(repair_features)
                    else np.empty(0, dtype="float64")
                )
                syndrome_probability = None
                repairs = select_graph_repairs(
                    repair_probabilities,
                    proposals,
                    threshold=float(repair_bundle["threshold"]),
                    maximum_additions=int(repair_bundle["maximum_additions"]),
                )
            sources = [
                (chunks[str(candidates[index]["chunk_id"])], float(support_probabilities[index]))
                for index in selected
                if str(candidates[index]["chunk_id"]) in chunks
            ]
            sources.extend(
                (chunks[str(proposals[index]["chunk_id"])], float(repair_probabilities[index]))
                for index in repairs
                if str(proposals[index]["chunk_id"]) in chunks
            )
            context, provenance = evidence_context(sources, args.context_character_budget)
            answer = reader.answer(example.question, context)
            score = answer_score(
                answer.prediction,
                str(example.answer),
                example.answer_aliases,
                dataset=dataset,
            )
            baseline = baseline_by_id[example.policy_id]
            record = {
                "format_version": FORMAT_VERSION,
                "id": example.policy_id,
                "dataset": dataset,
                "question": example.question,
                "reasoning_type": example.reasoning_type,
                "prediction": answer.prediction,
                "em": score.em,
                "f1": score.f1,
                "baseline_prediction": baseline["prediction"],
                "baseline_em": float(baseline["em"]),
                "baseline_f1": float(baseline["f1"]),
                "base_selected_indices": [int(index) for index in selected],
                "repair_indices": [int(index) for index in repairs],
                "repair_count": len(repairs),
                "syndrome_probability": syndrome_probability,
                "repair_evidence": [
                    {
                        "chunk_id": str(proposals[index]["chunk_id"]),
                        "content_id": str(proposals[index]["content_id"]),
                        "title": str(proposals[index]["title"]),
                        "probability": float(repair_probabilities[index]),
                    }
                    for index in repairs
                ],
                "evidence": list(provenance),
                "cache_hit": answer.cache_hit,
                "latency_seconds": answer.latency_seconds,
            }
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows_by_id[example.policy_id] = record
            new += 1
            completed = len(rows_by_id)
            if completed % args.log_every == 0 or completed == len(joined):
                current = list(rows_by_id.values())
                elapsed = __import__("time").perf_counter() - started
                eta = elapsed / max(1, new) * (len(joined) - completed)
                print(
                    f"{dataset} repair {completed}/{len(joined)} | ETA {_duration(eta)} | "
                    f"F1={np.mean([row['baseline_f1'] for row in current])*100:.2f}->"
                    f"{np.mean([row['f1'] for row in current])*100:.2f} | "
                    f"avg add={np.mean([row['repair_count'] for row in current]):.2f}",
                    flush=True,
                )
                _write_manifest(manifest, config, "in_progress", completed)
    _write_manifest(manifest, config, "ready", len(rows_by_id))
    return [rows_by_id[example.policy_id] for _, example in joined], retriever.model, retriever.reranker


def main() -> None:
    args = _arguments()
    support_model = joblib.load(args.support_model)["model"]
    width_bundle = joblib.load(args.width_model)
    repair_bundle = joblib.load(args.repair_model)
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
                support_model,
                width_bundle,
                repair_bundle,
                reader,
                reader_config,
                shared_model,
                shared_reranker,
            )
            print(
                f"{dataset}: n={len(rows)} topology EM/F1="
                f"{np.mean([row['baseline_em'] for row in rows])*100:.2f}/"
                f"{np.mean([row['baseline_f1'] for row in rows])*100:.2f} repair EM/F1="
                f"{np.mean([row['em'] for row in rows])*100:.2f}/"
                f"{np.mean([row['f1'] for row in rows])*100:.2f} additions="
                f"{np.mean([row['repair_count'] for row in rows]):.2f}",
                flush=True,
            )
    session.close()


if __name__ == "__main__":
    main()
