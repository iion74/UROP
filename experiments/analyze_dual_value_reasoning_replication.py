"""Combine development and disjoint holdout adaptive-reasoning pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicated adaptive-reasoning analysis")
    parser.add_argument(
        "--development-root",
        default="data/benchmarks/multihop/context_reader_aware/dual_value_choice_judge",
    )
    parser.add_argument(
        "--holdout-root",
        default="data/benchmarks/multihop/context_reader_aware/dual_value_choice_judge_holdout_seed43",
    )
    parser.add_argument(
        "--additional-root",
        action="append",
        default=[],
        help="추가 독립 실행 결과 root입니다. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--router-analysis",
        default="data/benchmarks/multihop/context_reader_aware/analysis/dual_value_router_multiseed.json",
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=(
            "data/benchmarks/multihop/context_reader_aware/analysis/"
            "dual_value_reasoning_replication.json"
        ),
    )
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.bootstrap_resamples < 1000:
        parser.error("bootstrap resamples는 1000 이상이어야 합니다.")
    return args


def _interval(values: np.ndarray, rng, resamples: int) -> dict[str, float]:
    samples = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return {
        "mean_points": float(values.mean() * 100.0),
        "ci95_low_points": float(np.quantile(samples, 0.025) * 100.0),
        "ci95_high_points": float(np.quantile(samples, 0.975) * 100.0),
    }


def _run_summary(rows: list[dict]) -> dict:
    em_delta = np.asarray(
        [float(row["judged_em"]) - float(row["baseline_em"]) for row in rows]
    )
    f1_delta = np.asarray(
        [float(row["judged_f1"]) - float(row["baseline_f1"]) for row in rows]
    )
    return {
        "count": len(rows),
        "em_change_points": float(em_delta.mean() * 100.0),
        "f1_change_points": float(f1_delta.mean() * 100.0),
    }


def main() -> None:
    args = _arguments()
    router = json.loads(Path(args.router_analysis).read_text(encoding="utf-8"))
    result = {
        "format_version": 1,
        "method": "dual_value_adaptive_reasoning_replicated_pilot",
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "datasets": {},
    }
    all_lower_bounds = []
    all_combined_positive = []
    for dataset_index, dataset in enumerate(args.datasets):
        named_roots = [
            ("development", Path(args.development_root)),
            ("disjoint_holdout", Path(args.holdout_root)),
            *[
                (f"additional_{index}", Path(root))
                for index, root in enumerate(args.additional_root, start=1)
            ],
        ]
        runs = {}
        for name, root in named_roots:
            path = root / f"{dataset}_validation.jsonl"
            if path.exists():
                runs[name] = list(_iter_jsonl(path))
            elif name in {"development", "disjoint_holdout"}:
                raise SystemExit(f"필수 실행 결과가 없습니다: {path}")
        seen_ids: set[str] = set()
        for name, run_rows in runs.items():
            run_ids = {str(row["id"]) for row in run_rows}
            overlap = seen_ids & run_ids
            if overlap:
                raise SystemExit(f"{dataset}: {name} ID가 선행 실행과 {len(overlap)}개 겹칩니다.")
            seen_ids.update(run_ids)
        rows = [row for run_rows in runs.values() for row in run_rows]
        rng = np.random.default_rng(args.seed + dataset_index)
        em_delta = np.asarray(
            [float(row["judged_em"]) - float(row["baseline_em"]) for row in rows]
        )
        f1_delta = np.asarray(
            [float(row["judged_f1"]) - float(row["baseline_f1"]) for row in rows]
        )
        em_interval = _interval(em_delta, rng, args.bootstrap_resamples)
        f1_interval = _interval(f1_delta, rng, args.bootstrap_resamples)
        baseline_em = float(np.mean([row["baseline_em"] for row in rows]))
        baseline_f1 = float(np.mean([row["baseline_f1"] for row in rows]))
        judged_em = float(np.mean([row["judged_em"] for row in rows]))
        judged_f1 = float(np.mean([row["judged_f1"] for row in rows]))
        thinking_latency = float(
            np.mean([float(row["repair_latency_seconds"]) for row in rows])
        )
        judge_latency = float(
            np.mean([float(row["judge_latency_seconds"]) for row in rows])
        )
        route = router["ensemble"][dataset]
        queue_rate = float(route["reason_queue_rate"])
        reason_queue_count = int(route["route_distribution"]["REASON"])
        reason_queue_coverage = len(rows) / reason_queue_count
        projected_em = float(route["em"]) + queue_rate * float(em_delta.mean())
        projected_f1 = float(route["f1"]) + queue_rate * float(f1_delta.mean())
        local = {
            "runs": {name: _run_summary(run_rows) for name, run_rows in runs.items()},
            "combined": {
                "count": len(rows),
                "baseline_em": baseline_em,
                "baseline_f1": baseline_f1,
                "judged_em": judged_em,
                "judged_f1": judged_f1,
                "em_delta": em_interval,
                "f1_delta": f1_interval,
                "wrong_to_exact": sum(
                    row["baseline_em"] < 1.0 and row["judged_em"] == 1.0 for row in rows
                ),
                "exact_to_wrong": sum(
                    row["baseline_em"] == 1.0 and row["judged_em"] < 1.0 for row in rows
                ),
                "selected_repair_rate": float(
                    np.mean([row["choice"] == "B" for row in rows])
                ),
                "average_thinking_latency_seconds": thinking_latency,
                "average_judge_latency_seconds": judge_latency,
                "average_total_extra_latency_seconds_when_reasoned": thinking_latency
                + judge_latency,
            },
            "projected_full_validation": {
                "status": (
                    "direct_measurement_of_complete_reason_queue"
                    if len(rows) == reason_queue_count
                    else "extrapolation_from_sampled_reason_queue"
                ),
                "evaluated_reason_questions": len(rows),
                "total_reason_queue_questions": reason_queue_count,
                "reason_queue_coverage": reason_queue_coverage,
                "reason_queue_rate": queue_rate,
                "router_pre_repair_em": float(route["em"]),
                "router_pre_repair_f1": float(route["f1"]),
                "projected_em": projected_em,
                "projected_f1": projected_f1,
                "projected_em_change_points_vs_fixed3": (
                    projected_em - float(route["fixed3_em"])
                )
                * 100.0,
                "projected_f1_change_points_vs_fixed3": (
                    projected_f1 - float(route["fixed3_f1"])
                )
                * 100.0,
                "projected_extra_latency_seconds_per_all_question": queue_rate
                * (thinking_latency + judge_latency),
            },
        }
        result["datasets"][dataset] = local
        all_lower_bounds.append(f1_interval["ci95_low_points"])
        all_combined_positive.append(f1_interval["mean_points"] > 0.0)
        print(
            f"{dataset}: combined ΔEM={em_interval['mean_points']:+.2f} "
            f"ΔF1={f1_interval['mean_points']:+.2f} "
            f"95% CI [{f1_interval['ci95_low_points']:+.2f}, "
            f"{f1_interval['ci95_high_points']:+.2f}] projected full F1="
            f"{projected_f1*100:.2f}",
            flush=True,
        )
    if all(bound > 0.0 for bound in all_lower_bounds):
        decision = "advance_to_full_reason_queue_evaluation"
    elif all(all_combined_positive):
        decision = "promising_but_requires_larger_disjoint_evaluation"
    else:
        decision = "reject_adaptive_reasoning"
    result["decision"] = decision
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"decision: {decision}\nsaved: {output}", flush=True)


if __name__ == "__main__":
    main()
