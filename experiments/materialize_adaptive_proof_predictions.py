"""Materialize evidence for a new threshold from cached prefix probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from experiments.evaluate_answer_oracle import _load_fixed_retriever
from experiments.train_adaptive_proof_completion_policy import ACTIONS, _candidate_list, _repair_selection
from experiments.train_counterfactual_proof_policy import _load_split
from experiments.train_graph_repair_policy import _base_selection
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _corpus_fingerprint


def _arguments():
    parser = argparse.ArgumentParser(description="Materialize thresholded adaptive proof evidence")
    parser.add_argument(
        "--source-predictions",
        default="data/models/adaptive_proof_completion_policy.report.predictions.jsonl",
    )
    parser.add_argument(
        "--policy-model",
        default="data/models/adaptive_proof_completion_policy_high_threshold.joblib",
    )
    parser.add_argument(
        "--trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--support-model", default="data/models/counterfactual_proof_completion_policy.joblib")
    parser.add_argument("--width-model", default="data/models/evidence_width_policy.joblib")
    parser.add_argument("--repair-model", default="data/models/factorized_graph_proof_repair_policy.joblib")
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-rows-per-dataset", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    return args


def main() -> None:
    args = _arguments()
    threshold = float(joblib.load(args.policy_model)["threshold"])
    support_model = joblib.load(args.support_model)["model"]
    width_bundle = joblib.load(args.width_model)
    repair_bundle = joblib.load(args.repair_model)
    cached = {
        str(row["id"]): row
        for row in map(json.loads, Path(args.source_predictions).read_text(encoding="utf-8").splitlines())
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shared_model = shared_reranker = None
    with output.open("w", encoding="utf-8") as destination:
        for dataset in args.datasets:
            joined, documents = _load_split(
                args,
                dataset,
                args.split,
                Path(args.trajectory_root),
                args.max_rows_per_dataset,
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
            for position, (trajectory, example) in enumerate(joined, start=1):
                source = cached[example.policy_id]
                probabilities = [float(value) for value in source["probabilities_by_step"]]
                step = next(
                    (index for index, probability in enumerate(probabilities) if probability >= threshold),
                    len(ACTIONS) - 1,
                )
                action = ACTIONS[step]
                candidates = _candidate_list(trajectory, action)
                selected, support_probabilities = _base_selection(
                    support_model, width_bundle, example.question, candidates, documents
                )
                proposals, rank_scores, _, repairs = _repair_selection(
                    repair_bundle,
                    example.question,
                    candidates,
                    selected,
                    support_probabilities,
                    documents,
                    retriever,
                )
                evidence = [
                    {
                        "chunk_id": str(candidates[index]["chunk_id"]),
                        "content_id": str(candidates[index]["content_id"]),
                        "title": str(candidates[index].get("title", "")),
                        "score": float(support_probabilities[index]),
                        "kind": "selected",
                    }
                    for index in selected
                ]
                evidence.extend(
                    {
                        "chunk_id": str(proposals[index]["chunk_id"]),
                        "content_id": str(proposals[index]["content_id"]),
                        "title": str(proposals[index].get("title", "")),
                        "score": float(rank_scores[index]),
                        "kind": "repair",
                    }
                    for index in repairs
                )
                row = {
                    **source,
                    "chosen_step": step,
                    "chosen_action": action,
                    "complete_probability": probabilities[step],
                    "evidence": evidence,
                    "materialized_threshold": threshold,
                }
                destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if position % args.log_every == 0 or position == len(joined):
                    print(f"{dataset} {position}/{len(joined)}", flush=True)
    print(f"saved: {output} threshold={threshold:.3f}", flush=True)


if __name__ == "__main__":
    main()
