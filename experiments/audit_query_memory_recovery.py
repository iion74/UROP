"""Audit whether persistent query memory recovers missing proof documents.

The frozen planner queries already stored in each trajectory are replayed
against the fixed corpus.  Their results are merged instead of evicting older
evidence, then the counterfactual proof scorer and topology-aware width policy
select the reader evidence.  Gold supports are used only for the audit metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import chain
from pathlib import Path

import joblib
import numpy as np

from experiments.evaluate_answer_oracle import _load_fixed_retriever
from experiments.train_counterfactual_proof_policy import (
    _candidates,
    _group_indices,
    _load_split,
    _support_metrics,
)
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _corpus_fingerprint
from src.causal_evidence import (
    evidence_width_features,
    proof_policy_features,
    select_topology_aware_indices,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent query-memory proof recovery audit")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument(
        "--trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--policy-model", default="data/models/counterfactual_proof_completion_policy.joblib")
    parser.add_argument("--width-model", default="data/models/evidence_width_policy.joblib")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-rows-per-dataset", type=int, default=100)
    parser.add_argument("--candidate-mode", choices=("graph", "query"), default="graph")
    parser.add_argument("--results-per-query", type=int, default=12)
    parser.add_argument("--graph-neighbors-per-seed", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(
        args.max_rows_per_dataset,
        args.results_per_query,
        args.graph_neighbors_per_seed,
        args.log_every,
    ) < 1:
        parser.error("rows/results/log 값은 1 이상이어야 합니다.")
    return args


def _planner_queries(row: dict, question: str) -> tuple[str, ...]:
    queries = [str(question).strip()]
    queries.extend(
        str(step.get("query", "")).strip()
        for step in row.get("planner_audit", [])
        if str(step.get("query", "")).strip()
    )
    return tuple(dict.fromkeys(query for query in queries if query))


def _merge_query_results(retriever, queries: tuple[str, ...], existing: list[dict], top_k: int) -> list[dict]:
    by_source: dict[str, dict] = {}
    for candidate in existing:
        source = str(candidate.get("content_id", ""))
        if not source:
            continue
        by_source[source] = {
            "chunk_id": str(candidate.get("chunk_id", "")),
            "content_id": source,
            "title": str(candidate.get("title", "")),
            "score": float(candidate.get("score", 0.0)),
            "query_hits": 0,
            "best_query_position": len(queries),
        }
    for query_position, query in enumerate(queries):
        for chunk, score in retriever.search(query, top_k=top_k):
            current = by_source.get(chunk.source)
            if current is None:
                current = {
                    "chunk_id": chunk.chunk_id,
                    "content_id": chunk.source,
                    "title": chunk.title,
                    "score": float(score),
                    "query_hits": 0,
                    "best_query_position": query_position,
                }
                by_source[chunk.source] = current
            current["query_hits"] += 1
            current["best_query_position"] = min(current["best_query_position"], query_position)
            if float(score) > float(current["score"]):
                current["chunk_id"] = chunk.chunk_id
                current["title"] = chunk.title
                current["score"] = float(score)
    return sorted(
        by_source.values(),
        key=lambda candidate: (
            -float(candidate["score"]),
            -int(candidate["query_hits"]),
            int(candidate["best_query_position"]),
            str(candidate["content_id"]),
        ),
    )


def _merge_graph_results(retriever, existing: list[dict], neighbors_per_seed: int) -> list[dict]:
    """Expand stored evidence over the prebuilt lightweight chunk graph."""
    by_source: dict[str, dict] = {}
    for candidate in existing:
        source = str(candidate.get("content_id", ""))
        if not source:
            continue
        by_source[source] = {
            "chunk_id": str(candidate.get("chunk_id", "")),
            "content_id": source,
            "title": str(candidate.get("title", "")),
            "score": float(candidate.get("score", 0.0)),
            "query_hits": 0,
            "best_query_position": 0,
        }
    for seed_position, candidate in enumerate(existing):
        chunk_position = retriever.chunk_positions.get(str(candidate.get("chunk_id", "")))
        if chunk_position is None:
            continue
        seed_score = float(candidate.get("score", 0.0))
        ranked_neighbors = sorted(
            retriever.graph.get(chunk_position, ()),
            key=lambda index: retriever._similarity(chunk_position, index),
            reverse=True,
        )[:neighbors_per_seed]
        for neighbor_position in ranked_neighbors:
            chunk = retriever.chunks[neighbor_position]
            similarity = (retriever._similarity(chunk_position, neighbor_position) + 1.0) / 2.0
            propagated_score = float(np.clip(0.55 * seed_score + 0.45 * similarity, 0.0, 1.0))
            current = by_source.get(chunk.source)
            if current is None:
                current = {
                    "chunk_id": chunk.chunk_id,
                    "content_id": chunk.source,
                    "title": chunk.title,
                    "score": propagated_score,
                    "query_hits": 1,
                    "best_query_position": seed_position + 1,
                }
                by_source[chunk.source] = current
            else:
                current["query_hits"] += 1
                current["best_query_position"] = min(
                    int(current["best_query_position"]), seed_position + 1
                )
                if propagated_score > float(current["score"]):
                    current["chunk_id"] = chunk.chunk_id
                    current["title"] = chunk.title
                    current["score"] = propagated_score
    return sorted(
        by_source.values(),
        key=lambda candidate: (
            -float(candidate["score"]),
            -int(candidate["query_hits"]),
            int(candidate["best_query_position"]),
            str(candidate["content_id"]),
        ),
    )


def _select(policy, width_bundle, question, candidates, documents) -> tuple[np.ndarray, int, np.ndarray]:
    matrix = proof_policy_features(question, candidates, documents, ())
    probabilities = policy.predict_proba(matrix)[:, 1][:-1]
    width_features = evidence_width_features(policy, question, candidates, documents)
    raw_width = float(width_bundle["model"].predict(width_features.reshape(1, -1))[0])
    width = int(
        np.clip(
            np.ceil(raw_width),
            int(width_bundle["minimum_width"]),
            min(int(width_bundle["maximum_width"]), len(candidates)),
        )
    )
    selected = np.asarray(
        select_topology_aware_indices(question, candidates, documents, probabilities, width),
        dtype="int64",
    )
    return selected, width, probabilities


def _dataset_report(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "queries": float(np.mean([row["query_count"] for row in rows])),
        "memory_documents": float(np.mean([row["memory_documents"] for row in rows])),
        "selected_documents": float(np.mean([row["selected_documents"] for row in rows])),
        "selected_width_distribution": dict(sorted(Counter(row["selected_documents"] for row in rows).items())),
        "original_candidate_all_support_recall": float(
            np.mean([row["original_candidate_all_support"] for row in rows]) * 100.0
        ),
        "original_selected_all_support_recall": float(
            np.mean([row["original_selected_all_support"] for row in rows]) * 100.0
        ),
        "memory_candidate_all_support_recall": float(
            np.mean([row["memory_candidate_all_support"] for row in rows]) * 100.0
        ),
        "memory_selected_all_support_recall": float(
            np.mean([row["memory_selected_all_support"] for row in rows]) * 100.0
        ),
        "memory_selected_source_recall": float(
            np.mean([row["memory_selected_source_recall"] for row in rows]) * 100.0
        ),
        "memory_selected_support_precision": float(
            np.mean([row["memory_selected_support_precision"] for row in rows]) * 100.0
        ),
    }


def main() -> None:
    args = _arguments()
    policy = joblib.load(args.policy_model)["model"]
    width_bundle = joblib.load(args.width_model)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    shared_model = shared_reranker = None
    full_report = {}
    for dataset in args.datasets:
        joined, documents = _load_split(
            args, dataset, args.split, Path(args.trajectory_root), args.max_rows_per_dataset
        )
        full_joined, _ = _load_split(args, dataset, args.split, Path(args.trajectory_root), 2000)
        full_examples = list({example.policy_id: example for _, example in full_joined}.values())
        retriever = _load_fixed_retriever(
            Path(args.index_root) / args.split / dataset,
            _corpus_fingerprint(full_examples),
            embedding_model=shared_model,
            reranker=shared_reranker,
        )
        shared_model = retriever.model or shared_model
        shared_reranker = retriever.reranker or shared_reranker
        records = []
        output = output_root / f"{dataset}_{args.split}.jsonl"
        with output.open("w", encoding="utf-8") as destination:
            for position, (row, example) in enumerate(joined, start=1):
                original = _candidates(row)
                groups = _group_indices(example, original)
                original_all = _support_metrics(set(range(len(original))), groups)
                original_selected, _, _ = _select(
                    policy, width_bundle, example.question, original, documents
                )
                original_selected_metrics = _support_metrics(set(original_selected.tolist()), groups)

                queries = _planner_queries(row, example.question)
                memory = (
                    _merge_graph_results(retriever, original, args.graph_neighbors_per_seed)
                    if args.candidate_mode == "graph"
                    else _merge_query_results(retriever, queries, original, args.results_per_query)
                )
                memory_groups = _group_indices(example, memory)
                memory_all = _support_metrics(set(range(len(memory))), memory_groups)
                selected, predicted_width, probabilities = _select(
                    policy, width_bundle, example.question, memory, documents
                )
                selected_metrics = _support_metrics(set(selected.tolist()), memory_groups)
                support_indices = set(chain.from_iterable(memory_groups))
                record = {
                    "id": example.policy_id,
                    "dataset": dataset,
                    "reasoning_type": example.reasoning_type,
                    "query_count": len(queries),
                    "queries": list(queries),
                    "memory_documents": len(memory),
                    "predicted_width": predicted_width,
                    "selected_documents": len(selected),
                    "selected": [
                        {
                            **memory[index],
                            "support_probability": float(probabilities[index]),
                            "is_gold_support": bool(index in support_indices),
                        }
                        for index in selected
                    ],
                    "original_candidate_all_support": original_all[1],
                    "original_selected_all_support": original_selected_metrics[1],
                    "memory_candidate_all_support": memory_all[1],
                    "memory_selected_source_recall": selected_metrics[0],
                    "memory_selected_all_support": selected_metrics[1],
                    "memory_selected_support_precision": selected_metrics[2],
                }
                records.append(record)
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                if position % args.log_every == 0 or position == len(joined):
                    partial = _dataset_report(records)
                    print(
                        f"{dataset} {position}/{len(joined)}: original/memory selected all-support="
                        f"{partial['original_selected_all_support_recall']:.2f}/"
                        f"{partial['memory_selected_all_support_recall']:.2f}",
                        flush=True,
                    )
        full_report[dataset] = _dataset_report(records)
        print(json.dumps({dataset: full_report[dataset]}, ensure_ascii=False, indent=2), flush=True)
    (output_root / "report.json").write_text(
        json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
