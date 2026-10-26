"""Aggregate reader-free or single-reader student policies across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analyze_context_reader_aware_multiseed import _paired_interval
from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Student policy multi-seed analysis")
    parser.add_argument("--prefix-root", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.bootstrap_samples < 1:
        parser.error("bootstrap samples는 1 이상이어야 합니다.")
    return args


def _planner_elapsed(prefix: dict, chosen_step: int) -> float:
    if chosen_step == 0:
        return 0.0
    actions = prefix["actions"]
    selected = actions[ACTIONS[chosen_step]]
    planner_total = float(prefix.get("planner_total_latency_seconds", 0.0))
    rewrite_total = float(
        actions[ACTIONS[-1]].get("planner_rewrite_latency_cumulative_seconds") or 0.0
    )
    selected_rewrite = selected.get("planner_rewrite_latency_cumulative_seconds")
    if selected_rewrite is None or rewrite_total <= 0.0:
        return planner_total * chosen_step / max(1, len(ACTIONS) - 1)
    retrieval_and_overhead = max(0.0, planner_total - rewrite_total)
    return float(selected_rewrite) + retrieval_and_overhead * chosen_step / max(
        1, len(ACTIONS) - 1
    )


def _costs(prefix: dict, chosen_step: int, method: str) -> dict[str, float]:
    actions = prefix["actions"]
    initial = float(prefix.get("initial_retrieval_latency_seconds", 0.0))

    def latency(step: int) -> float:
        return float(actions[ACTIONS[step]].get("reader_latency_seconds") or 0.0)

    def tokens(step: int) -> float:
        return float(actions[ACTIONS[step]].get("reader_generated_tokens") or 0.0)

    fixed_latency = initial + float(prefix.get("planner_total_latency_seconds", 0.0)) + latency(3)
    if method in {"reader_free_retrieval_student", "reader_free_evidence_graph_student"}:
        adaptive_latency = initial + _planner_elapsed(prefix, chosen_step) + latency(chosen_step)
        adaptive_tokens = tokens(chosen_step)
        reader_calls = 1.0
    elif method == "single_reader_selective_student":
        adaptive_latency = initial + _planner_elapsed(prefix, chosen_step) + latency(2)
        adaptive_tokens = tokens(2)
        reader_calls = 1.0
        if chosen_step == 3:
            adaptive_latency += latency(3)
            adaptive_tokens += tokens(3)
            reader_calls += 1.0
    else:
        raise ValueError(f"지원하지 않는 student method: {method}")
    return {
        "fixed_latency": fixed_latency,
        "adaptive_latency": adaptive_latency,
        "fixed_tokens": tokens(3),
        "adaptive_tokens": adaptive_tokens,
        "reader_calls": reader_calls,
    }


def main() -> None:
    args = _arguments()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    methods = {str(report["method"]) for report in reports}
    if len(methods) != 1:
        raise SystemExit(f"서로 다른 student method를 한 번에 분석할 수 없습니다: {sorted(methods)}")
    method = methods.pop()
    seeds = [str(report["training_sample_seed"]) for report in reports]
    predictions_by_seed = []
    for report in reports:
        rows = list(_iter_jsonl(Path(report["validation_predictions"])))
        predictions_by_seed.append({str(row["id"]): row for row in rows})

    dataset_results = {}
    for dataset_index, dataset in enumerate(args.datasets):
        prefix_rows = list(_iter_jsonl(Path(args.prefix_root) / f"{dataset}_validation.jsonl"))
        prefixes = {str(row["id"]): row for row in prefix_rows}
        ids = list(prefixes)
        if not ids:
            raise SystemExit(f"{dataset}: prefix가 없습니다.")
        fixed_em = np.asarray([float(prefixes[row_id]["actions"][ACTIONS[3]]["em"]) for row_id in ids])
        fixed_f1 = np.asarray([float(prefixes[row_id]["actions"][ACTIONS[3]]["f1"]) for row_id in ids])
        seed_rows = []
        em_vectors = []
        f1_vectors = []
        continue_vectors = []
        fixed_latency_reference = None
        fixed_tokens_reference = None
        for seed, predictions in zip(seeds, predictions_by_seed, strict=True):
            missing = [row_id for row_id in ids if row_id not in predictions]
            if missing:
                raise SystemExit(f"{dataset}/seed={seed}: prediction이 {len(missing):,}개 부족합니다.")
            rows = [predictions[row_id] for row_id in ids]
            adaptive_em = np.asarray([float(row["em"]) for row in rows])
            adaptive_f1 = np.asarray([float(row["f1"]) for row in rows])
            searches = np.asarray([float(row["chosen_step"]) for row in rows])
            costs = [
                _costs(prefixes[row_id], int(predictions[row_id]["chosen_step"]), method)
                for row_id in ids
            ]
            fixed_latency = float(np.mean([item["fixed_latency"] for item in costs]))
            adaptive_latency = float(np.mean([item["adaptive_latency"] for item in costs]))
            fixed_tokens = float(np.mean([item["fixed_tokens"] for item in costs]))
            adaptive_tokens = float(np.mean([item["adaptive_tokens"] for item in costs]))
            fixed_latency_reference = fixed_latency
            fixed_tokens_reference = fixed_tokens
            seed_rows.append(
                {
                    "training_sample_seed": seed,
                    "em": float(adaptive_em.mean()),
                    "f1": float(adaptive_f1.mean()),
                    "em_change_vs_fixed3_points": float((adaptive_em - fixed_em).mean() * 100.0),
                    "f1_change_vs_fixed3_points": float((adaptive_f1 - fixed_f1).mean() * 100.0),
                    "average_additional_searches": float(searches.mean()),
                    "search_reduction_percent_vs_fixed3": float((3.0 - searches.mean()) / 3.0 * 100.0),
                    "estimated_online_latency_seconds": adaptive_latency,
                    "latency_reduction_percent_vs_fixed3": float((fixed_latency - adaptive_latency) / fixed_latency * 100.0),
                    "average_reader_generated_tokens": adaptive_tokens,
                    "reader_token_reduction_percent_vs_fixed3": float((fixed_tokens - adaptive_tokens) / fixed_tokens * 100.0),
                    "average_reader_calls": float(np.mean([item["reader_calls"] for item in costs])),
                }
            )
            em_vectors.append(adaptive_em)
            f1_vectors.append(adaptive_f1)
            continue_vectors.append(searches == 3.0)
        em_mean = np.vstack(em_vectors).mean(axis=0)
        f1_mean = np.vstack(f1_vectors).mean(axis=0)
        continue_votes = np.vstack(continue_vectors).sum(axis=0)
        ensemble_masks = {
            "unanimous_stop": continue_votes >= 1,
            "majority_vote": continue_votes >= (len(continue_vectors) // 2 + 1),
            "any_stop": continue_votes == len(continue_vectors),
        }
        ensemble_results = {}
        for rule_index, (rule, continue_mask) in enumerate(ensemble_masks.items()):
            chosen_steps = np.where(continue_mask, 3, 2)
            ensemble_em = np.asarray(
                [
                    float(prefixes[row_id]["actions"][ACTIONS[int(step)]]["em"])
                    for row_id, step in zip(ids, chosen_steps, strict=True)
                ]
            )
            ensemble_f1 = np.asarray(
                [
                    float(prefixes[row_id]["actions"][ACTIONS[int(step)]]["f1"])
                    for row_id, step in zip(ids, chosen_steps, strict=True)
                ]
            )
            ensemble_costs = [
                _costs(prefixes[row_id], int(step), method)
                for row_id, step in zip(ids, chosen_steps, strict=True)
            ]
            adaptive_latency = float(
                np.mean([item["adaptive_latency"] for item in ensemble_costs])
            )
            adaptive_tokens = float(
                np.mean([item["adaptive_tokens"] for item in ensemble_costs])
            )
            average_searches = float(chosen_steps.mean())
            ensemble_results[rule] = {
                "em": float(ensemble_em.mean()),
                "f1": float(ensemble_f1.mean()),
                "em_change_vs_fixed3": _paired_interval(
                    ensemble_em - fixed_em,
                    args.bootstrap_samples,
                    args.seed + dataset_index * 100 + rule_index * 2 + 20,
                ),
                "f1_change_vs_fixed3": _paired_interval(
                    ensemble_f1 - fixed_f1,
                    args.bootstrap_samples,
                    args.seed + dataset_index * 100 + rule_index * 2 + 21,
                ),
                "average_additional_searches": average_searches,
                "search_reduction_percent_vs_fixed3": float(
                    (3.0 - average_searches) / 3.0 * 100.0
                ),
                "estimated_online_latency_seconds": adaptive_latency,
                "latency_reduction_percent_vs_fixed3": float(
                    (fixed_latency_reference - adaptive_latency)
                    / fixed_latency_reference
                    * 100.0
                ),
                "average_reader_generated_tokens": adaptive_tokens,
            }
        dataset_results[dataset] = {
            "count": len(ids),
            "fixed3": {
                "em": float(fixed_em.mean()),
                "f1": float(fixed_f1.mean()),
                "average_additional_searches": 3.0,
                "estimated_online_latency_seconds": fixed_latency_reference,
                "average_reader_generated_tokens": fixed_tokens_reference,
            },
            "student_across_training_seeds": {
                "seed_count": len(seed_rows),
                "em_mean": float(np.mean([row["em"] for row in seed_rows])),
                "em_std": float(np.std([row["em"] for row in seed_rows], ddof=1)) if len(seed_rows) > 1 else 0.0,
                "f1_mean": float(np.mean([row["f1"] for row in seed_rows])),
                "f1_std": float(np.std([row["f1"] for row in seed_rows], ddof=1)) if len(seed_rows) > 1 else 0.0,
                "average_additional_searches_mean": float(np.mean([row["average_additional_searches"] for row in seed_rows])),
                "estimated_online_latency_seconds_mean": float(np.mean([row["estimated_online_latency_seconds"] for row in seed_rows])),
                "average_reader_generated_tokens_mean": float(np.mean([row["average_reader_generated_tokens"] for row in seed_rows])),
                "per_seed": seed_rows,
            },
            "comparison_vs_fixed3": {
                "em": _paired_interval(em_mean - fixed_em, args.bootstrap_samples, args.seed + dataset_index * 10),
                "f1": _paired_interval(f1_mean - fixed_f1, args.bootstrap_samples, args.seed + dataset_index * 10 + 1),
            },
            "deployment_ensembles": ensemble_results,
        }

    payload = {
        "format_version": 1,
        "method": method,
        "training_sample_seeds": seeds,
        "bootstrap_samples": args.bootstrap_samples,
        "cost_note": "Reader token accounting excludes planner rewrite tokens because legacy prefix rows did not store them. Latency uses exact cumulative rewrite time plus a proportional allocation of retrieval/overhead residual; step3 exactly equals recorded total planner latency. Reader-free methods call the reader exactly once.",
        "datasets": dataset_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dataset_results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
