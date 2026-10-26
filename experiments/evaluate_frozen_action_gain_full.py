"""Evaluate frozen, directly learned action gains on all validation questions.

SEARCH and REASON heads were fitted on ID-disjoint train questions in earlier
experiments.  This script does not refit or tune either head on validation.
It requires a judged REASON counterfactual for *every* validation question,
so matched-budget random routing can be measured without extrapolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments.analyze_student_policy_multiseed import _planner_elapsed
from experiments.evaluate_answer_gain_search_policy import (
    DATASETS,
    ROOT,
    _paired_interval,
    _read_dataset,
)
from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_reader_free_student_policy import policy_feature_names


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-root", type=Path, default=ROOT / "full_validation")
    parser.add_argument(
        "--judge-root", type=Path,
        default=ROOT / "analysis" / "action_gain_full_validation_judge",
    )
    parser.add_argument(
        "--reason-pilot-root", type=Path,
        default=ROOT / "analysis" / "action_gain_full_validation_reason",
    )
    parser.add_argument(
        "--reason-train-judge-root", type=Path,
        default=ROOT / "analysis" / "disjoint_holdout_seed20260916_500" / "all_reason_judge",
    )
    parser.add_argument(
        "--old-predictions", type=Path,
        default=ROOT / "analysis" / "dual_value_full_system.predictions.jsonl",
    )
    parser.add_argument(
        "--search-model", type=Path,
        default=Path("data/models/answer_gain_search_probe_reader_aware.joblib"),
    )
    parser.add_argument(
        "--reason-model", type=Path,
        default=Path("data/models/reason_gain_gate_reverse.joblib"),
    )
    parser.add_argument(
        "--search-train-root", type=Path, default=ROOT / "train_pool_seed42_2000",
    )
    parser.add_argument(
        "--reason-train-root", type=Path, default=ROOT / "disjoint_holdout_seed20260916_500",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "analysis" / "frozen_action_gain_full_validation",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--random-draws", type=int, default=5000)
    return parser.parse_args()


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_unique(path: Path) -> dict[str, dict]:
    rows = list(_jsonl(path))
    indexed = {str(row["id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return indexed


def _prefix_extra(path: Path, ids: list[str]) -> dict[str, np.ndarray]:
    rows = _index_unique(path)
    if set(rows) != set(ids):
        raise ValueError(f"Prefix IDs differ: {path}")
    result = {}
    for step in range(4):
        actions = [rows[identifier]["actions"][ACTIONS[step]] for identifier in ids]
        result[f"f1_{step}"] = np.asarray([float(action["f1"]) for action in actions])
        result[f"em_{step}"] = np.asarray([float(action["em"]) for action in actions])
        result[f"tokens_{step}"] = np.asarray([
            float(action.get("reader_generated_tokens") or 0) for action in actions
        ])
        result[f"latency_{step}"] = np.asarray([
            float(rows[identifier].get("initial_retrieval_latency_seconds", 0))
            + _planner_elapsed(rows[identifier], step)
            + float(action.get("reader_latency_seconds") or 0)
            for identifier, action in zip(ids, actions, strict=True)
        ])
    return result


def _reason_outcomes(path: Path, ids: list[str], step2_f1, step2_em) -> dict[str, np.ndarray]:
    judged = _index_unique(path)
    if set(judged) != set(ids):
        missing = set(ids) - set(judged)
        extra = set(judged) - set(ids)
        raise ValueError(f"Incomplete REASON outcomes: {path}; missing={len(missing)}, extra={len(extra)}")
    ordered = [judged[identifier] for identifier in ids]
    baseline_f1 = np.asarray([float(row["baseline_f1"]) for row in ordered])
    baseline_em = np.asarray([float(row["baseline_em"]) for row in ordered])
    if not np.allclose(baseline_f1, step2_f1, atol=1e-12, rtol=0) or not np.allclose(
        baseline_em, step2_em, atol=1e-12, rtol=0
    ):
        raise ValueError(f"Judged REASON baselines do not match prefix step 2: {path}")
    return {
        "f1": np.asarray([float(row["judged_f1"]) for row in ordered]),
        "em": np.asarray([float(row["judged_em"]) for row in ordered]),
        "extra_latency": np.asarray([
            float(row["repair_latency_seconds"]) + float(row["judge_latency_seconds"])
            for row in ordered
        ]),
        "extra_tokens": np.asarray([
            float(row.get("repair_generated_tokens") or 0)
            + float(row.get("judge_generated_tokens") or 0)
            for row in ordered
        ]),
        "extra_reader_calls": np.asarray([
            1.0 if row.get("judge_skipped_identical", False) else 2.0
            for row in ordered
        ]),
    }


def _scores(routes: np.ndarray, data: dict, reason: dict) -> dict[str, np.ndarray]:
    search = routes == "SEARCH"
    think = routes == "REASON"
    return {
        "f1": np.where(search, data["f1_3"], np.where(think, reason["f1"], data["f1_2"])),
        "em": np.where(search, data["em_3"], np.where(think, reason["em"], data["em_2"])),
        # Every route first produces the step-2 answer, including SEARCH.
        "latency": np.where(
            search, data["latency_3"] + data["reader_2_latency"],
            data["latency_2"] + np.where(think, reason["extra_latency"], 0),
        ),
        "tokens": np.where(
            search, data["tokens_3"] + data["tokens_2"],
            data["tokens_2"] + np.where(think, reason["extra_tokens"], 0),
        ),
    }


def _route_from_gains(search_gain: np.ndarray, reason_gain: np.ndarray) -> np.ndarray:
    if search_gain.shape != reason_gain.shape:
        raise ValueError("SEARCH and REASON score shapes differ")
    # A zero-gain tie favors STOP; a SEARCH/REASON tie favors cheaper SEARCH.
    return np.where(
        (search_gain > 0) & (search_gain >= reason_gain), "SEARCH",
        np.where(reason_gain > 0, "REASON", "STOP"),
    )


def _compare(
    values: np.ndarray, reference: np.ndarray, samples: int, seed: int,
    metric: str = "f1",
) -> dict:
    difference = values - reference
    return {
        f"{metric}_change_points": float(difference.mean() * 100),
        f"{metric}_change_ci95": _paired_interval(difference, samples, seed),
    }


def _random_permutation_p(
    data: dict, reason: dict, search_count: int, reason_count: int,
    observed_f1: float, draws: int, seed: int,
) -> float:
    n = len(data["ids"])
    rng = np.random.default_rng(seed)
    gain_search = data["f1_3"] - data["f1_2"]
    gain_reason = reason["f1"] - data["f1_2"]
    baseline_sum = float(data["f1_2"].sum())
    at_least = 0
    for _ in range(draws):
        order = rng.permutation(n)
        actual = (
            baseline_sum + float(gain_search[order[:search_count]].sum())
            + float(gain_reason[order[search_count:search_count + reason_count]].sum())
        ) / n
        at_least += actual >= observed_f1 - 1e-12
    return at_least / draws


def _policy_report(
    routes: np.ndarray, data: dict, reason: dict, old: dict,
    *, bootstrap: int, draws: int, seed: int,
) -> dict:
    n = len(routes)
    outcomes = _scores(routes, data, reason)
    counts = {route: int(np.sum(routes == route)) for route in ("STOP", "SEARCH", "REASON")}
    search_rate = counts["SEARCH"] / n
    reason_rate = counts["REASON"] / n
    stop_rate = counts["STOP"] / n
    expected_random = (
        stop_rate * data["f1_2"]
        + search_rate * data["f1_3"]
        + reason_rate * reason["f1"]
    )
    expected_random_em = (
        stop_rate * data["em_2"]
        + search_rate * data["em_3"]
        + reason_rate * reason["em"]
    )
    report = {
        "count": n,
        "routes": counts,
        "average_additional_searches": float(2 + search_rate),
        "average_reader_calls": float(
            1 + search_rate + np.mean((routes == "REASON") * reason["extra_reader_calls"])
        ),
        "em": float(outcomes["em"].mean() * 100),
        "f1": float(outcomes["f1"].mean() * 100),
        "estimated_latency_seconds": float(outcomes["latency"].mean()),
        "reader_generated_tokens": float(outcomes["tokens"].mean()),
        "vs_fixed3": _compare(outcomes["f1"], data["f1_3"], bootstrap, seed),
        "vs_previous_policy": _compare(outcomes["f1"], old["f1"], bootstrap, seed + 1),
        "vs_fixed3_em": _compare(outcomes["em"], data["em_3"], bootstrap, seed + 4, "em"),
        "vs_previous_policy_em": _compare(outcomes["em"], old["em"], bootstrap, seed + 5, "em"),
        "matched_random_expected_f1": float(expected_random.mean() * 100),
        "matched_random_expected_em": float(expected_random_em.mean() * 100),
        "vs_matched_random": {
            **_compare(outcomes["f1"], expected_random, bootstrap, seed + 2),
            "em_change_points": float((outcomes["em"] - expected_random_em).mean() * 100),
            "em_change_ci95": _paired_interval(
                outcomes["em"] - expected_random_em, bootstrap, seed + 6
            ),
            "fraction_random_at_least_policy": _random_permutation_p(
                data, reason, counts["SEARCH"], counts["REASON"],
                float(outcomes["f1"].mean()), draws, seed + 3,
            ),
        },
    }
    return report


def _fixed_report(data: dict, step: int) -> dict:
    return {
        "em": float(data[f"em_{step}"].mean() * 100),
        "f1": float(data[f"f1_{step}"].mean() * 100),
        "average_additional_searches": step,
        "average_reader_calls": 1,
        "estimated_latency_seconds": float(data[f"latency_{step}"].mean()),
        "reader_generated_tokens": float(data[f"tokens_{step}"].mean()),
    }


def main() -> None:
    args = _arguments()
    if args.bootstrap_samples < 1 or args.random_draws < 1:
        raise ValueError("Bootstrap samples and random draws must be positive")
    manifest = json.loads((args.prefix_root / "hotpotqa_validation.manifest.json").read_text())
    pilot_summary = json.loads((args.reason_pilot_root / "summary.json").read_text())
    judge_summary = json.loads((args.judge_root / "summary.json").read_text())
    reason_train_summary = json.loads((args.reason_train_judge_root / "summary.json").read_text())
    digests = {
        manifest["reader"]["model_digest"],
        pilot_summary["reader"]["model_digest"],
        judge_summary["judge"]["model_digest"],
        reason_train_summary["judge"]["model_digest"],
    }
    if len(digests) != 1:
        raise ValueError("Qwen model digests differ across train, prefix, REASON, or judge")
    if any(reason_train_summary["datasets"][dataset]["count"] != 500 for dataset in DATASETS):
        raise ValueError("REASON training counterfactual coverage is incomplete")
    names = tuple(manifest["feature_names"])
    expected_names = policy_feature_names(names, "single_reader")
    search_bundle = joblib.load(args.search_model)
    reason_bundle = joblib.load(args.reason_model)
    for label, bundle in (("SEARCH", search_bundle), ("REASON", reason_bundle)):
        if tuple(bundle["feature_names"]) != expected_names:
            raise ValueError(f"{label} model feature schema is incompatible with prefix data")
    old_rows = _index_unique(args.old_predictions)
    search_train_ids = {
        dataset: set(_index_unique(args.search_train_root / f"{dataset}_train.jsonl"))
        for dataset in DATASETS
    }
    reason_train_ids = {
        dataset: set(_index_unique(args.reason_train_root / f"{dataset}_train.jsonl"))
        for dataset in DATASETS
    }
    if any(len(search_train_ids[dataset]) != 2000 or len(reason_train_ids[dataset]) != 500
           for dataset in DATASETS):
        raise ValueError("Unexpected training question counts")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "frozen_direct_answer_f1_gain_three_action_policy",
        "decision": "after_step2_argmax_of_STOP_0_SEARCH_predicted_gain_REASON_predicted_gain; ties_choose_cheaper_action",
        "model_hashes_sha256": {
            "search": _sha256(args.search_model),
            "reason": _sha256(args.reason_model),
        },
        "models": {
            "search": {
                "class": type(search_bundle["model"]).__name__,
                "parameters": search_bundle["model"].get_params(),
            },
            "reason": {
                "class": type(reason_bundle["model"]).__name__,
                "parameters": reason_bundle["model"].get_params(),
            },
            "feature_count": len(expected_names),
        },
        "training": {
            "search": "6000 ID-disjoint train questions (2000 per dataset), actual F1 at step3 minus step2",
            "reason": "1500 ID-disjoint train questions (500 per dataset), actual judged F1 minus step2",
            "generation_model": "frozen qwen3:8b; only shallow value heads were trained",
            "generation_model_digest": next(iter(digests)),
            "validation_tuning": False,
        },
        "evaluation": {
            "split": "complete official validation questions in cached prefix files",
            "datasets": list(DATASETS),
            "all_action_counterfactuals_required": True,
            "bootstrap_samples": args.bootstrap_samples,
            "matched_random_draws": args.random_draws,
            "latency": "sum of cached retrieval/reader/action components; not live end-to-end timing",
            "identical_candidate_short_circuit": "judge call is omitted only when baseline and repair answer strings are exactly equal; answer scores are unchanged",
            "caveat": "Official validation was inspected during earlier exploratory work; this is not a new untouched test set.",
        },
        "datasets": {},
    }
    for index, dataset in enumerate(DATASETS):
        data = _read_dataset(args.prefix_root, dataset, "validation", names, "single_reader")
        if set(data["ids"]) & (search_train_ids[dataset] | reason_train_ids[dataset]):
            raise ValueError(f"Training/evaluation question ID overlap: {dataset}")
        data.update(_prefix_extra(args.prefix_root / f"{dataset}_validation.jsonl", data["ids"]))
        reason = _reason_outcomes(
            args.judge_root / f"{dataset}_validation.jsonl",
            data["ids"], data["f1_2"], data["em_2"],
        )
        old = [old_rows[identifier] for identifier in data["ids"]]
        old_arrays = {
            key: np.asarray([float(row[key]) for row in old])
            for key in ("f1", "em", "latency_seconds", "reader_generated_tokens", "searches", "reader_calls")
        }
        if any(row["dataset"] != dataset for row in old):
            raise ValueError(f"Old prediction dataset mismatch: {dataset}")
        search_gain = search_bundle["model"].predict(data["x"])
        reason_gain = reason_bundle["model"].predict(data["x"])
        joint = _route_from_gains(search_gain, reason_gain)
        search_only = np.where(search_gain > 0, "SEARCH", "STOP")
        reason_only = np.where(reason_gain > 0, "REASON", "STOP")
        dataset_report = {
            "count": len(data["ids"]),
            "fixed": {str(step): _fixed_report(data, step) for step in range(4)},
            "previous_policy": {
                "em": float(old_arrays["em"].mean() * 100),
                "f1": float(old_arrays["f1"].mean() * 100),
                "average_additional_searches": float(old_arrays["searches"].mean()),
                "average_reader_calls": float(old_arrays["reader_calls"].mean()),
                "estimated_latency_seconds": float(old_arrays["latency_seconds"].mean()),
                "reader_generated_tokens": float(old_arrays["reader_generated_tokens"].mean()),
            },
            "policies": {},
            "all_reason_oracle_f1": float(np.maximum(data["f1_2"], np.maximum(data["f1_3"], reason["f1"])).mean() * 100),
        }
        for variant, routes in (("joint", joint), ("search_only", search_only), ("reason_only", reason_only)):
            dataset_report["policies"][variant] = _policy_report(
                routes, data, reason, old_arrays,
                bootstrap=args.bootstrap_samples, draws=args.random_draws,
                seed=100 + 10 * index + ("joint", "search_only", "reason_only").index(variant),
            )
        report["datasets"][dataset] = dataset_report
        joint_outcomes = _scores(joint, data, reason)
        decisions = args.output_root / f"{dataset}_validation.decisions.jsonl"
        with decisions.open("w", encoding="utf-8") as handle:
            for i, identifier in enumerate(data["ids"]):
                handle.write(json.dumps({
                    "id": identifier,
                    "dataset": dataset,
                    "predicted_search_gain": float(search_gain[i]),
                    "predicted_reason_gain": float(reason_gain[i]),
                    "route": str(joint[i]),
                    "f1": float(joint_outcomes["f1"][i]),
                    "em": float(joint_outcomes["em"][i]),
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
        joint_result = dataset_report["policies"]["joint"]
        print(
            f"{dataset}: n={len(data['ids'])}, joint EM={joint_result['em']:.3f}, "
            f"F1={joint_result['f1']:.3f}, fixed3={dataset_report['fixed']['3']['f1']:.3f}, "
            f"old={dataset_report['previous_policy']['f1']:.3f}, "
            f"random={joint_result['matched_random_expected_f1']:.3f}, "
            f"routes={joint_result['routes']}", flush=True,
        )
    if len(old_rows) != sum(item["count"] for item in report["datasets"].values()):
        raise ValueError("Old-policy predictions do not match full evaluation size")
    path = args.output_root / "summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
