"""Materialise and score the complete dual-value adaptive RAG system."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.analyze_context_reader_aware_multiseed import _paired_interval
from experiments.analyze_student_policy_multiseed import _planner_elapsed
from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl
from experiments.train_reader_free_student_policy import DECISION_STEP, FINAL_STEP


SEARCH = "SEARCH"
REASON = "REASON"
STOP = "STOP"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete dual-value system analysis")
    parser.add_argument(
        "--prefix-root",
        default="data/benchmarks/multihop/context_reader_aware/full_validation",
    )
    parser.add_argument("--model-root", default="data/models/dual_value_router")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument(
        "--router-predictions",
        default="",
        help="고정된 ensemble의 별도 평가 split 예측. 생략하면 기존 validation seed 예측을 사용합니다.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--judge-root", action="append", required=True)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/benchmarks/multihop/context_reader_aware/analysis/dual_value_full_system.json",
    )
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    args.seeds = tuple(dict.fromkeys(args.seeds))
    if len(args.seeds) < 3 or len(args.seeds) % 2 == 0:
        parser.error("ensemble seed는 3개 이상의 홀수여야 합니다.")
    if args.bootstrap_samples < 1000:
        parser.error("bootstrap samples는 1000 이상이어야 합니다.")
    return args


def _seed_predictions(model_root: Path, seeds: tuple[int, ...], predictions_path: str = "") -> list[dict[str, dict]]:
    if predictions_path:
        rows = list(_iter_jsonl(Path(predictions_path)))
        by_id = {str(row["id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise SystemExit(f"동결 router prediction ID가 중복되었습니다: {predictions_path}")
        return [by_id]
    result = []
    for seed in seeds:
        report_path = model_root / f"policy_seed{seed}.report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = list(_iter_jsonl(Path(str(report["validation_predictions"]))))
        result.append({str(row["id"]): row for row in rows})
    return result


def _judged_rows(roots: list[str], dataset: str, split: str = "validation") -> dict[str, dict]:
    result = {}
    for root in roots:
        path = Path(root) / f"{dataset}_{split}.jsonl"
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            identifier = str(row["id"])
            if identifier in result:
                raise SystemExit(f"{dataset}: 중재 결과 ID가 중복됩니다: {identifier}")
            result[identifier] = row
    return result


def _ensemble_route(identifier: str, predictions: list[dict[str, dict]]) -> tuple[str, dict]:
    if len(predictions) == 1:
        row = predictions[0][identifier]
        return str(row["route"]), dict(row.get("route_votes", {}))
    votes = Counter(seed_rows[identifier]["route"] for seed_rows in predictions)
    majority = len(predictions) // 2 + 1
    if votes[SEARCH] >= majority:
        route = SEARCH
    elif votes[REASON] > votes[STOP]:
        route = REASON
    else:
        route = STOP
    return route, dict(votes)


def _number(action: dict, key: str) -> float:
    return float(action.get(key) or 0.0)


def _record(prefix: dict, route: str, votes: dict, judged: dict | None) -> dict:
    chosen_step = FINAL_STEP if route == SEARCH else DECISION_STEP
    action = prefix["actions"][ACTIONS[chosen_step]]
    direct = prefix["actions"][ACTIONS[0]]
    fixed = prefix["actions"][ACTIONS[FINAL_STEP]]
    em = _number(action, "em")
    f1 = _number(action, "f1")
    prediction = str(action["prediction"])
    thinking_em = em
    thinking_f1 = f1
    extra_latency = 0.0
    extra_tokens = 0.0
    reader_calls = 1.0
    if route == REASON:
        if judged is None:
            raise SystemExit(f"REASON 중재 결과가 없습니다: {prefix['id']}")
        em = _number(judged, "judged_em")
        f1 = _number(judged, "judged_f1")
        prediction = str(judged["judged_prediction"])
        thinking_em = _number(judged, "repair_em")
        thinking_f1 = _number(judged, "repair_f1")
        extra_latency = _number(judged, "repair_latency_seconds") + _number(
            judged, "judge_latency_seconds"
        )
        extra_tokens = _number(judged, "repair_generated_tokens") + _number(
            judged, "judge_generated_tokens"
        )
        reader_calls += 2.0
    initial = float(prefix.get("initial_retrieval_latency_seconds", 0.0))
    adaptive_latency = (
        initial
        + _planner_elapsed(prefix, chosen_step)
        + _number(action, "reader_latency_seconds")
        + extra_latency
    )
    fixed_latency = (
        initial
        + float(prefix.get("planner_total_latency_seconds", 0.0))
        + _number(fixed, "reader_latency_seconds")
    )
    direct_latency = initial + _number(direct, "reader_latency_seconds")
    return {
        "id": str(prefix["id"]),
        "dataset": str(prefix["dataset"]),
        "route": route,
        "route_votes": votes,
        "prediction": prediction,
        "em": em,
        "f1": f1,
        "thinking_only_em": thinking_em,
        "thinking_only_f1": thinking_f1,
        "direct_em": _number(direct, "em"),
        "direct_f1": _number(direct, "f1"),
        "fixed3_em": _number(fixed, "em"),
        "fixed3_f1": _number(fixed, "f1"),
        "searches": float(chosen_step),
        "reader_calls": reader_calls,
        "latency_seconds": adaptive_latency,
        "direct_latency_seconds": direct_latency,
        "fixed3_latency_seconds": fixed_latency,
        "reader_generated_tokens": _number(action, "reader_generated_tokens") + extra_tokens,
        "fixed3_reader_generated_tokens": _number(fixed, "reader_generated_tokens"),
    }


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _dataset_summary(rows: list[dict], bootstrap_samples: int, seed: int) -> dict:
    em = np.asarray([row["em"] for row in rows], dtype="float64")
    f1 = np.asarray([row["f1"] for row in rows], dtype="float64")
    direct_em = np.asarray([row["direct_em"] for row in rows], dtype="float64")
    direct_f1 = np.asarray([row["direct_f1"] for row in rows], dtype="float64")
    fixed_em = np.asarray([row["fixed3_em"] for row in rows], dtype="float64")
    fixed_f1 = np.asarray([row["fixed3_f1"] for row in rows], dtype="float64")
    thinking_em = np.asarray([row["thinking_only_em"] for row in rows], dtype="float64")
    thinking_f1 = np.asarray([row["thinking_only_f1"] for row in rows], dtype="float64")
    routes = Counter(str(row["route"]) for row in rows)
    latency = _mean(rows, "latency_seconds")
    fixed_latency = _mean(rows, "fixed3_latency_seconds")
    tokens = _mean(rows, "reader_generated_tokens")
    fixed_tokens = _mean(rows, "fixed3_reader_generated_tokens")
    return {
        "count": len(rows),
        "em": float(em.mean()),
        "f1": float(f1.mean()),
        "direct_em": float(direct_em.mean()),
        "direct_f1": float(direct_f1.mean()),
        "fixed3_em": float(fixed_em.mean()),
        "fixed3_f1": float(fixed_f1.mean()),
        "thinking_only_em": float(thinking_em.mean()),
        "thinking_only_f1": float(thinking_f1.mean()),
        "em_vs_fixed3": _paired_interval(em - fixed_em, bootstrap_samples, seed),
        "f1_vs_fixed3": _paired_interval(f1 - fixed_f1, bootstrap_samples, seed + 1),
        "em_vs_direct": _paired_interval(em - direct_em, bootstrap_samples, seed + 2),
        "f1_vs_direct": _paired_interval(f1 - direct_f1, bootstrap_samples, seed + 3),
        "thinking_only_em_vs_fixed3": _paired_interval(
            thinking_em - fixed_em, bootstrap_samples, seed + 4
        ),
        "thinking_only_f1_vs_fixed3": _paired_interval(
            thinking_f1 - fixed_f1, bootstrap_samples, seed + 5
        ),
        "arbitration_em_gain": _paired_interval(
            em - thinking_em, bootstrap_samples, seed + 6
        ),
        "arbitration_f1_gain": _paired_interval(
            f1 - thinking_f1, bootstrap_samples, seed + 7
        ),
        "route_distribution": dict(routes),
        "average_searches": _mean(rows, "searches"),
        "search_reduction_percent_vs_fixed3": (3.0 - _mean(rows, "searches")) / 3.0 * 100.0,
        "average_reader_calls": _mean(rows, "reader_calls"),
        "estimated_latency_seconds": latency,
        "fixed3_latency_seconds": fixed_latency,
        "latency_change_percent_vs_fixed3": (latency - fixed_latency) / fixed_latency * 100.0,
        "average_reader_generated_tokens": tokens,
        "fixed3_reader_generated_tokens": fixed_tokens,
        "reader_token_change_percent_vs_fixed3": (tokens - fixed_tokens) / fixed_tokens * 100.0,
    }


def main() -> None:
    args = _arguments()
    predictions = _seed_predictions(Path(args.model_root), args.seeds, args.router_predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_output = output.with_name(f"{output.stem}.predictions.jsonl")
    summaries = {}
    all_records = []
    for dataset_index, dataset in enumerate(args.datasets):
        judged = _judged_rows(args.judge_root, dataset, args.split)
        prefixes = list(_iter_jsonl(Path(args.prefix_root) / f"{dataset}_{args.split}.jsonl"))
        records = []
        reason_ids = set()
        for prefix in prefixes:
            identifier = str(prefix["id"])
            route, votes = _ensemble_route(identifier, predictions)
            if route == REASON:
                reason_ids.add(identifier)
            records.append(_record(prefix, route, votes, judged.get(identifier)))
        missing = reason_ids - set(judged)
        extra = set(judged) - reason_ids
        if missing or extra:
            raise SystemExit(
                f"{dataset}: REASON 중재 coverage 오류 missing={len(missing)}, extra={len(extra)}"
            )
        summaries[dataset] = _dataset_summary(
            records, args.bootstrap_samples, args.seed + dataset_index * 10
        )
        all_records.extend(records)
        item = summaries[dataset]
        print(
            f"{dataset}: n={item['count']} EM/F1={item['em']*100:.2f}/{item['f1']*100:.2f} "
            f"vs fixed3={item['em_vs_fixed3']['observed_change_points']:+.2f}/"
            f"{item['f1_vs_fixed3']['observed_change_points']:+.2f} "
            f"search={item['average_searches']:.2f} latency={item['estimated_latency_seconds']:.2f}s",
            flush=True,
        )
    with prediction_output.open("w", encoding="utf-8") as destination:
        for row in all_records:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    result = {
        "format_version": 1,
        "method": "dual_value_search_reason_stop_with_thinking_arbitration",
        "seeds": list(args.seeds),
        "validation_predictions": str(prediction_output),
        "judge_roots": args.judge_root,
        "datasets": summaries,
        "evaluation_scope": (
            "complete_official_question_local_validation"
            if args.split == "validation"
            else "sampled_disjoint_train_holdout"
        ),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output}\npredictions: {prediction_output}", flush=True)


if __name__ == "__main__":
    main()
