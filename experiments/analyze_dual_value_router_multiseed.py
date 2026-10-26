"""Aggregate dual-value router runs without selecting a favourable seed."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl
from experiments.train_reader_free_student_policy import DECISION_STEP, FINAL_STEP


SEARCH = "SEARCH"
REASON = "REASON"
STOP = "STOP"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five-seed dual-value router analysis")
    parser.add_argument("--model-root", default="data/models/dual_value_router")
    parser.add_argument(
        "--validation-root",
        default="data/benchmarks/multihop/context_reader_aware/full_validation",
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--output",
        default="data/benchmarks/multihop/context_reader_aware/analysis/dual_value_router_multiseed.json",
    )
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    args.seeds = tuple(dict.fromkeys(args.seeds))
    if len(args.seeds) < 3 or len(args.seeds) % 2 == 0:
        parser.error("majority ensemble에는 3개 이상의 홀수 시드가 필요합니다.")
    return args


def _load_seed(model_root: Path, seed: int) -> tuple[dict, dict[str, dict]]:
    report_path = model_root / f"policy_seed{seed}.report.json"
    if not report_path.exists():
        raise SystemExit(f"report가 없습니다: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prediction_path = Path(str(report["validation_predictions"]))
    if not prediction_path.exists():
        raise SystemExit(f"prediction이 없습니다: {prediction_path}")
    predictions = {str(row["id"]): row for row in _iter_jsonl(prediction_path)}
    return report, predictions


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype="float64")
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _ensemble_dataset(rows: list[dict], predictions: list[dict[str, dict]]) -> tuple[dict, list[dict]]:
    majority = len(predictions) // 2 + 1
    em = []
    f1 = []
    fixed_em = []
    fixed_f1 = []
    routes = Counter()
    vote_patterns = Counter()
    reason_error = 0
    reason_records = []
    for row in rows:
        identifier = str(row["id"])
        seed_routes = [seed_predictions[identifier]["route"] for seed_predictions in predictions]
        votes = Counter(seed_routes)
        if votes[SEARCH] >= majority:
            route = SEARCH
        elif votes[REASON] > votes[STOP]:
            route = REASON
        else:
            route = STOP
        chosen_step = FINAL_STEP if route == SEARCH else DECISION_STEP
        chosen = row["actions"][ACTIONS[chosen_step]]
        fixed = row["actions"][ACTIONS[FINAL_STEP]]
        em.append(float(chosen["em"]))
        f1.append(float(chosen["f1"]))
        fixed_em.append(float(fixed["em"]))
        fixed_f1.append(float(fixed["f1"]))
        routes[route] += 1
        vote_patterns["/".join(str(votes[action]) for action in (SEARCH, REASON, STOP))] += 1
        if route == REASON:
            is_error = float(row["actions"][ACTIONS[DECISION_STEP]]["em"]) < 1.0
            reason_error += int(is_error)
            reason_records.append(
                {
                    "id": identifier,
                    "dataset": str(row["dataset"]),
                    "question": str(row["question"]),
                    "route_votes": dict(votes),
                    "baseline_prediction": str(chosen["prediction"]),
                    "baseline_em": float(chosen["em"]),
                    "baseline_f1": float(chosen["f1"]),
                }
            )
    count = len(rows)
    em_mean = float(np.mean(em))
    f1_mean = float(np.mean(f1))
    fixed_em_mean = float(np.mean(fixed_em))
    fixed_f1_mean = float(np.mean(fixed_f1))
    report = {
        "count": count,
        "em": em_mean,
        "f1": f1_mean,
        "fixed3_em": fixed_em_mean,
        "fixed3_f1": fixed_f1_mean,
        "em_change_points": (em_mean - fixed_em_mean) * 100.0,
        "f1_change_points": (f1_mean - fixed_f1_mean) * 100.0,
        "average_searches": 2.0 + routes[SEARCH] / count,
        "search_reduction_percent_vs_fixed3": (1.0 - routes[SEARCH] / count) / 3.0 * 100.0,
        "route_distribution": dict(routes),
        "reason_queue_rate": routes[REASON] / count,
        "answer_error_precision_in_reason_queue": reason_error / max(1, routes[REASON]),
        "vote_patterns_search_reason_stop": dict(vote_patterns),
    }
    return report, reason_records


def main() -> None:
    args = _arguments()
    model_root = Path(args.model_root)
    loaded = [_load_seed(model_root, seed) for seed in args.seeds]
    reports = [item[0] for item in loaded]
    predictions = [item[1] for item in loaded]
    result = {
        "format_version": 1,
        "method": "dual_value_router_five_seed_majority",
        "seeds": list(args.seeds),
        "selection_policy": (
            "SEARCH when it has a strict majority; otherwise choose REASON over STOP "
            "only when REASON has more votes"
        ),
        "per_seed": {},
        "ensemble": {},
    }
    reason_records = []
    validation_root = Path(args.validation_root)
    for dataset in args.datasets:
        result["per_seed"][dataset] = {
            metric: _mean_std(
                [
                    float(report["validation_pre_repair"][dataset][metric])
                    for report in reports
                ]
            )
            for metric in (
                "em",
                "f1",
                "average_additional_searches",
                "search_reduction_percent_vs_fixed3",
                "reason_queue_rate",
                "answer_error_precision_in_reason_queue",
            )
        }
        rows = list(_iter_jsonl(validation_root / f"{dataset}_validation.jsonl"))
        ensemble, local_reason = _ensemble_dataset(rows, predictions)
        result["ensemble"][dataset] = ensemble
        reason_records.extend(local_reason)
        print(
            f"{dataset}: ensemble routes={ensemble['route_distribution']} "
            f"EM/F1={ensemble['em']*100:.2f}/{ensemble['f1']*100:.2f} "
            f"delta={ensemble['em_change_points']:+.2f}/{ensemble['f1_change_points']:+.2f} "
            f"search={ensemble['average_searches']:.2f}",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reason_path = output.with_name(f"{output.stem}.reason_queue.jsonl")
    result["reason_queue"] = str(reason_path)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with reason_path.open("w", encoding="utf-8") as destination:
        for row in reason_records:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"saved: {output}\nreason queue: {reason_path}", flush=True)


if __name__ == "__main__":
    main()
