"""Aggregate full-validation reader-aware policies across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-context reader-aware multi-seed analysis")
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


def _paired_interval(differences: np.ndarray, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype="float64")
    for start in range(0, samples, 128):
        end = min(samples, start + 128)
        indices = rng.integers(0, len(differences), size=(end - start, len(differences)))
        draws[start:end] = differences[indices].mean(axis=1) * 100.0
    return {
        "observed_change_points": float(differences.mean() * 100.0),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0.0)),
        "probability_noninferior_margin_1_point": float(np.mean(draws >= -1.0)),
    }


def _mean(rows, key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _estimated_latencies(prefix: dict, chosen_step: int) -> tuple[float, float, float]:
    initial = float(prefix.get("initial_retrieval_latency_seconds", 0.0))
    planner_total = float(prefix.get("planner_total_latency_seconds", 0.0))
    actions = prefix["actions"]
    direct = initial + float(actions[ACTIONS[0]].get("reader_latency_seconds", 0.0))
    fixed3 = initial + planner_total + float(
        actions[ACTIONS[-1]].get("reader_latency_seconds", 0.0)
    )
    selected = actions[ACTIONS[chosen_step]]
    rewrite_total = float(
        actions[ACTIONS[-1]].get("planner_rewrite_latency_cumulative_seconds") or 0.0
    )
    selected_rewrite = selected.get("planner_rewrite_latency_cumulative_seconds")
    if selected_rewrite is None or rewrite_total <= 0.0:
        planner_elapsed = planner_total * chosen_step / max(1, len(ACTIONS) - 1)
    else:
        retrieval_and_overhead = max(0.0, planner_total - rewrite_total)
        planner_elapsed = float(selected_rewrite) + retrieval_and_overhead * chosen_step / max(
            1, len(ACTIONS) - 1
        )
    adaptive = initial + planner_elapsed + sum(
        float(actions[ACTIONS[step]].get("reader_latency_seconds", 0.0))
        for step in range(chosen_step + 1)
    )
    return direct, fixed3, adaptive


def _tokens(prefix: dict, chosen_step: int) -> tuple[float, float, float]:
    actions = prefix["actions"]
    def generated(step: int) -> int:
        return int(actions[ACTIONS[step]].get("reader_generated_tokens") or 0)
    return float(generated(0)), float(generated(len(ACTIONS) - 1)), float(
        sum(generated(step) for step in range(chosen_step + 1))
    )


def main() -> None:
    args = _arguments()
    report_payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    seed_predictions = []
    seed_labels = []
    for report in report_payloads:
        predictions_path = Path(report["validation_predictions"])
        predictions = list(_iter_jsonl(predictions_path))
        seed_predictions.append({str(row["id"]): row for row in predictions})
        seed_labels.append(str(report.get("training_sample_seed", report.get("seed", len(seed_labels)))))

    result = {}
    for dataset_index, dataset in enumerate(args.datasets):
        prefix_rows = list(
            _iter_jsonl(Path(args.prefix_root) / f"{dataset}_validation.jsonl")
        )
        prefixes = {str(row["id"]): row for row in prefix_rows}
        ids = list(prefixes)
        if not ids:
            raise SystemExit(f"{dataset}: 전체 validation prefix가 없습니다.")
        predictions_by_seed = []
        for label, predictions in zip(seed_labels, seed_predictions, strict=True):
            subset = {row_id: predictions[row_id] for row_id in ids if row_id in predictions}
            if subset.keys() != prefixes.keys():
                raise SystemExit(f"{dataset}/seed={label}: prediction ID가 전체 prefix와 다릅니다.")
            predictions_by_seed.append(subset)

        direct_em = np.asarray(
            [float(prefixes[row_id]["actions"][ACTIONS[0]]["em"]) for row_id in ids]
        )
        direct_f1 = np.asarray(
            [float(prefixes[row_id]["actions"][ACTIONS[0]]["f1"]) for row_id in ids]
        )
        fixed_em = np.asarray(
            [float(prefixes[row_id]["actions"][ACTIONS[-1]]["em"]) for row_id in ids]
        )
        fixed_f1 = np.asarray(
            [float(prefixes[row_id]["actions"][ACTIONS[-1]]["f1"]) for row_id in ids]
        )
        seed_results = []
        adaptive_em_vectors = []
        adaptive_f1_vectors = []
        for label, predictions in zip(seed_labels, predictions_by_seed, strict=True):
            rows = [predictions[row_id] for row_id in ids]
            adaptive_em = np.asarray([float(row["em"]) for row in rows])
            adaptive_f1 = np.asarray([float(row["f1"]) for row in rows])
            searches = np.asarray([float(row["chosen_step"]) for row in rows])
            latency_rows = [
                _estimated_latencies(prefixes[row_id], int(predictions[row_id]["chosen_step"]))
                for row_id in ids
            ]
            token_rows = [
                _tokens(prefixes[row_id], int(predictions[row_id]["chosen_step"]))
                for row_id in ids
            ]
            seed_results.append(
                {
                    "training_sample_seed": label,
                    "em": float(adaptive_em.mean()),
                    "f1": float(adaptive_f1.mean()),
                    "em_change_vs_fixed3_points": float((adaptive_em - fixed_em).mean() * 100.0),
                    "f1_change_vs_fixed3_points": float((adaptive_f1 - fixed_f1).mean() * 100.0),
                    "average_additional_searches": float(searches.mean()),
                    "search_reduction_percent_vs_fixed3": float((3.0 - searches.mean()) / 3.0 * 100.0),
                    "estimated_online_latency_seconds": float(np.mean([row[2] for row in latency_rows])),
                    "average_reader_generated_tokens": float(np.mean([row[2] for row in token_rows])),
                }
            )
            adaptive_em_vectors.append(adaptive_em)
            adaptive_f1_vectors.append(adaptive_f1)

        adaptive_em_matrix = np.vstack(adaptive_em_vectors)
        adaptive_f1_matrix = np.vstack(adaptive_f1_vectors)
        per_question_mean_em = adaptive_em_matrix.mean(axis=0)
        per_question_mean_f1 = adaptive_f1_matrix.mean(axis=0)
        fixed_latency = []
        direct_latency = []
        fixed_tokens = []
        direct_tokens = []
        for row_id in ids:
            d_latency, f_latency, _ = _estimated_latencies(prefixes[row_id], 3)
            d_tokens, f_tokens, _ = _tokens(prefixes[row_id], 3)
            direct_latency.append(d_latency)
            fixed_latency.append(f_latency)
            direct_tokens.append(d_tokens)
            fixed_tokens.append(f_tokens)

        result[dataset] = {
            "count": len(ids),
            "direct_chain_rag": {
                "em": float(direct_em.mean()),
                "f1": float(direct_f1.mean()),
                "average_additional_searches": 0.0,
                "estimated_online_latency_seconds": float(np.mean(direct_latency)),
                "average_reader_generated_tokens": float(np.mean(direct_tokens)),
            },
            "chain_fixed3": {
                "em": float(fixed_em.mean()),
                "f1": float(fixed_f1.mean()),
                "average_additional_searches": 3.0,
                "estimated_online_latency_seconds": float(np.mean(fixed_latency)),
                "average_reader_generated_tokens": float(np.mean(fixed_tokens)),
            },
            "adaptive_across_training_seeds": {
                "seed_count": len(seed_results),
                "em_mean": float(np.mean([row["em"] for row in seed_results])),
                "em_std": float(np.std([row["em"] for row in seed_results], ddof=1)) if len(seed_results) > 1 else 0.0,
                "f1_mean": float(np.mean([row["f1"] for row in seed_results])),
                "f1_std": float(np.std([row["f1"] for row in seed_results], ddof=1)) if len(seed_results) > 1 else 0.0,
                "average_additional_searches_mean": float(np.mean([row["average_additional_searches"] for row in seed_results])),
                "average_additional_searches_std": float(np.std([row["average_additional_searches"] for row in seed_results], ddof=1)) if len(seed_results) > 1 else 0.0,
                "estimated_online_latency_seconds_mean": float(np.mean([row["estimated_online_latency_seconds"] for row in seed_results])),
                "per_seed": seed_results,
            },
            "comparisons": {
                "adaptive_seed_mean_vs_fixed3": {
                    "em": _paired_interval(
                        per_question_mean_em - fixed_em,
                        args.bootstrap_samples,
                        args.seed + dataset_index * 10,
                    ),
                    "f1": _paired_interval(
                        per_question_mean_f1 - fixed_f1,
                        args.bootstrap_samples,
                        args.seed + dataset_index * 10 + 1,
                    ),
                },
                "adaptive_seed_mean_vs_direct": {
                    "em": _paired_interval(
                        per_question_mean_em - direct_em,
                        args.bootstrap_samples,
                        args.seed + dataset_index * 10 + 2,
                    ),
                    "f1": _paired_interval(
                        per_question_mean_f1 - direct_f1,
                        args.bootstrap_samples,
                        args.seed + dataset_index * 10 + 3,
                    ),
                },
            },
        }

    payload = {
        "format_version": 1,
        "method": "official_context_reader_aware_multiseed_full_validation",
        "training_seed_count": len(seed_labels),
        "training_sample_seeds": seed_labels,
        "bootstrap_samples": args.bootstrap_samples,
        "latency_note": "Estimated online latency uses exact cumulative rewrite latency plus a proportional allocation of retrieval/overhead residual; the terminal step exactly matches recorded total planner latency. Legacy rows use a proportional fallback.",
        "datasets": result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
