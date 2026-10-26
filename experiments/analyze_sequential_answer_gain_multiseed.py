"""Audit and aggregate independent train-sample-seed policy evaluations.

Each input report must have used a fresh 2,000-question sample from each of
the three train datasets and the same complete validation prefixes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from experiments.evaluate_answer_gain_search_policy import DATASETS, ROOT


EXPECTED_COUNTS = {"hotpotqa": 7405, "musique": 2417, "2wiki": 12576}
MEASURES = ("f1", "em", "average_total_searches", "delta_f1_points",
            "delta_em_points", "delta_searches", "search_reduction_percent",
            "estimated_latency_seconds", "reader_generated_tokens",
            "delta_estimated_latency_seconds", "delta_reader_generated_tokens")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "analysis" / "sequential_answer_gain_multiseed.json")
    args = parser.parse_args()
    if len(args.reports) < 3:
        parser.error("At least three independent training-sample seeds are required")
    return args


def _aggregate(reports: list[dict]) -> dict:
    seeds = [report["training_sample_seed"] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate training sample seed")
    reference = reports[0]
    for report in reports:
        if report["method"] != "sequential_best_future_f1_gain_hgb":
            raise ValueError("Incompatible policy method")
        if report["training_questions"] != 6000 or report["training_states"] != 18000:
            raise ValueError("Expected 2,000 questions per dataset and three train states each")
        if (report["feature_count"] != reference["feature_count"] or
                report["search_cost_f1_fraction"] != reference["search_cost_f1_fraction"]):
            raise ValueError("Policy structure or training settings changed across seeds")
        if report["validation_root"] != reference["validation_root"]:
            raise ValueError("Different validation roots")
        if set(report["training_cohort_by_dataset"]) != set(DATASETS):
            raise ValueError("Missing training dataset")
        for dataset in DATASETS:
            if report["training_cohort_by_dataset"][dataset]["questions"] != 2000:
                raise ValueError(f"{dataset}: expected 2,000 train questions")
    for dataset in DATASETS:
        digests = [report["training_cohort_by_dataset"][dataset]["selected_ids_sha256"]
                   for report in reports]
        if len(set(digests)) != len(digests):
            raise ValueError(f"{dataset}: the same training sample was reused")

    result = {
        "method": reference["method"],
        "training_sample_seeds": seeds,
        "training_questions_per_dataset_per_seed": 2000,
        "validation_root": reference["validation_root"],
        "feature_count": reference["feature_count"],
        "search_cost_f1_fraction": reference["search_cost_f1_fraction"],
        "datasets": {},
        "notes": [
            "All scores are percentage points out of 100; search counts include the mandatory first search.",
            "Standard deviations use the sample formula (ddof=1) across independent train-sample seeds.",
            "These are cached-trajectory replay results, not measured online end-to-end latency.",
            "The full validation splits were used during method development; they are not untouched test sets.",
        ],
    }
    for dataset in DATASETS:
        rows = [report["evaluations"]["full_validation"][dataset] for report in reports]
        if any(row["questions"] != EXPECTED_COUNTS[dataset] for row in rows):
            raise ValueError(f"{dataset}: incomplete validation evaluation")
        fixed = rows[0]["fixed"]["4"]
        fixed_cost = rows[0]["fixed4_single_reader"]
        if any(any(abs(row["fixed"]["4"][metric] - fixed[metric]) > 1e-9
                       for metric in ("f1", "em")) for row in rows):
            raise ValueError(f"{dataset}: fixed-four baseline changed across reports")
        if any(any(abs(row["fixed4_single_reader"][metric] - fixed_cost[metric]) > 1e-9
                       for metric in ("estimated_latency_seconds", "reader_generated_tokens"))
               for row in rows):
            raise ValueError(f"{dataset}: fixed-four cost baseline changed across reports")
        seed_rows = []
        for seed, report, row in zip(seeds, reports, rows, strict=True):
            policy = row["policy"]
            searches = policy["average_total_searches"]
            seed_rows.append({
                "training_sample_seed": seed,
                "training_root": report["training_root"],
                "training_ids_sha256": report["training_cohort_by_dataset"][dataset]["selected_ids_sha256"],
                "f1": policy["f1"],
                "em": policy["em"],
                "average_total_searches": searches,
                "delta_f1_points": policy["f1"] - fixed["f1"],
                "delta_em_points": policy["em"] - fixed["em"],
                "delta_searches": searches - 4.0,
                "search_reduction_percent": (4.0 - searches) / 4.0 * 100.0,
                "estimated_latency_seconds": policy["estimated_latency_seconds"],
                "reader_generated_tokens": policy["reader_generated_tokens"],
                "delta_estimated_latency_seconds": (
                    policy["estimated_latency_seconds"] - fixed_cost["estimated_latency_seconds"]
                ),
                "delta_reader_generated_tokens": (
                    policy["reader_generated_tokens"] - fixed_cost["reader_generated_tokens"]
                ),
            })
        result["datasets"][dataset] = {
            "questions": EXPECTED_COUNTS[dataset],
            "fixed4": {**fixed, **fixed_cost, "average_total_searches": 4.0},
            "per_seed": seed_rows,
            "across_seeds": {
                metric: {
                    "mean": mean(row[metric] for row in seed_rows),
                    "sample_std": stdev(row[metric] for row in seed_rows),
                }
                for metric in MEASURES
            },
        }
    return result


def main() -> None:
    args = _arguments()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    result = _aggregate(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for dataset, values in result["datasets"].items():
        metrics = values["across_seeds"]
        print(f"{dataset} ({values['questions']:,}): "
              f"F1 {metrics['f1']['mean']:.3f} ± {metrics['f1']['sample_std']:.3f}, "
              f"EM {metrics['em']['mean']:.3f} ± {metrics['em']['sample_std']:.3f}, "
              f"searches {metrics['average_total_searches']['mean']:.3f} ± "
              f"{metrics['average_total_searches']['sample_std']:.3f}, "
              f"ΔF1 vs fixed4 {metrics['delta_f1_points']['mean']:+.3f} ± "
              f"{metrics['delta_f1_points']['sample_std']:.3f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
