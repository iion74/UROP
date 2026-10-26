"""Evaluate frozen answer-gain SEARCH heads with the existing REASON head.

Uses the already generated all-REASON counterfactuals on the 500-question
ID-disjoint holdout.  The answer-error head is not retrained.  No Qwen calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from experiments.evaluate_answer_gain_search_policy import (
    DATASETS,
    ROOT,
    _paired_interval,
    _read_dataset,
)


HOLDOUT = ROOT / "disjoint_holdout_seed20260916_500"
ANALYSIS = ROOT / "analysis" / "disjoint_holdout_seed20260916_500"
OUTPUT = ROOT / "analysis" / "answer_gain_three_way_holdout.json"


def _rows(path: Path) -> dict[str, dict]:
    with path.open() as handle:
        result = {str(row["id"]): row for line in handle if (row := json.loads(line))}
    return result


def _compare(values: np.ndarray, baseline: np.ndarray, *, seed: int) -> dict:
    difference = values - baseline
    return {
        "change_points": float(difference.mean() * 100),
        "ci95": _paired_interval(difference, 3000, seed),
    }


def _random_three_way_fraction(
    step2: np.ndarray, step3: np.ndarray, reason: np.ndarray,
    routes: np.ndarray, observed: float, seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(routes)
    search_count = int(np.sum(routes == "SEARCH"))
    reason_count = int(np.sum(routes == "REASON"))
    at_least = 0
    draws = 5000
    for _ in range(draws):
        order = rng.permutation(n)
        random_f1 = step2.copy()
        random_f1[order[:search_count]] = step3[order[:search_count]]
        random_f1[order[search_count:search_count + reason_count]] = reason[
            order[search_count:search_count + reason_count]
        ]
        at_least += float(random_f1.mean()) >= observed - 1e-12
    return at_least / draws


def main() -> None:
    manifest = json.loads((HOLDOUT / "hotpotqa_train.manifest.json").read_text())
    names = tuple(manifest["feature_names"])
    bundles = [joblib.load(Path("data/models/dual_value_router") / f"policy_seed{s}.joblib")
               for s in range(42, 47)]
    search_bundles = {
        "reader_free": joblib.load("data/models/answer_gain_search_probe.joblib"),
        "reader_aware": joblib.load("data/models/answer_gain_search_probe_reader_aware.joblib"),
    }
    old_rows = _rows(ANALYSIS / "full_system.predictions.jsonl")
    report = {
        "scope": "500 ID-disjoint train questions per dataset; cached counterfactual evaluation",
        "method": "frozen direct-F1-gain SEARCH head plus unchanged answer-error REASON heads",
        "notes": [
            "The new SEARCH heads were trained on a different 2000-question-per-dataset pool.",
            "REASON still predicts current EM error, not the true benefit of reasoning.",
            "Reader-aware SEARCH requires an additional step-2 reader call before the step-3 answer.",
            "Latencies are estimates from cached timings, not live end-to-end measurements.",
            "This probe followed inspection of earlier holdout results and is exploratory.",
        ],
        "datasets": {},
    }
    for dataset in DATASETS:
        data_reader = _read_dataset(HOLDOUT, dataset, "train", names, "single_reader")
        data_free = _read_dataset(HOLDOUT, dataset, "train", names, "retrieval_only")
        if data_reader["ids"] != data_free["ids"]:
            raise ValueError("Feature modes changed question order")
        ids = data_reader["ids"]
        judge_rows = _rows(ANALYSIS / "all_reason_judge" / f"{dataset}_train.jsonl")
        if set(ids) != set(judge_rows):
            raise ValueError(f"Missing all-REASON counterfactuals: {dataset}")
        old_f1 = np.asarray([float(old_rows[identifier]["f1"]) for identifier in ids])
        old_em = np.asarray([float(old_rows[identifier]["em"]) for identifier in ids])
        old_latency = np.asarray([float(old_rows[identifier]["latency_seconds"]) for identifier in ids])
        old_tokens = np.asarray([float(old_rows[identifier]["reader_generated_tokens"]) for identifier in ids])
        old_reader_calls = np.asarray([float(old_rows[identifier]["reader_calls"]) for identifier in ids])
        old_searches = np.asarray([float(old_rows[identifier]["searches"]) for identifier in ids])
        old_routes = np.asarray([str(old_rows[identifier]["route"]) for identifier in ids])
        reason_f1 = np.asarray([float(judge_rows[identifier]["judged_f1"]) for identifier in ids])
        reason_em = np.asarray([float(judge_rows[identifier]["judged_em"]) for identifier in ids])
        reason_extra_latency = np.asarray([
            float(judge_rows[identifier]["repair_latency_seconds"])
            + float(judge_rows[identifier]["judge_latency_seconds"])
            for identifier in ids
        ])
        reason_extra_tokens = np.asarray([
            float(judge_rows[identifier]["repair_generated_tokens"])
            + float(judge_rows[identifier]["judge_generated_tokens"])
            for identifier in ids
        ])
        for identifier in ids:
            if old_rows[identifier]["route"] == "REASON" and abs(
                float(old_rows[identifier]["f1"]) - float(judge_rows[identifier]["judged_f1"])
            ) > 1e-12:
                raise ValueError(f"Old and all-REASON judged outputs differ: {identifier}")
        votes = np.zeros(len(ids), dtype=int)
        for bundle in bundles:
            probability = bundle["answer_model"].predict_proba(data_reader["x"])[:, 1]
            votes += probability >= float(bundle["answer_threshold"])
        item = {
            "count": len(ids),
            "fixed2_f1": float(data_reader["f1_2"].mean() * 100),
            "fixed3_f1": float(data_reader["f1_3"].mean() * 100),
            "old_full_f1": float(old_f1.mean() * 100),
            "old_full_em": float(old_em.mean() * 100),
            "old_full_estimated_latency_seconds": float(old_latency.mean()),
            "old_full_reader_generated_tokens": float(old_tokens.mean()),
            "old_full_average_reader_calls": float(old_reader_calls.mean()),
            "old_full_average_additional_searches": float(old_searches.mean()),
            "variants": {},
        }
        for mode, bundle in search_bundles.items():
            data = data_free if mode == "reader_free" else data_reader
            score = bundle["model"].predict(data["x"])
            search = score > float(bundle["pure_threshold"])
            reason = (~search) & (votes >= 3)
            routes = np.where(search, "SEARCH", np.where(reason, "REASON", "STOP"))
            f1 = np.where(search, data["f1_3"], np.where(reason, reason_f1, data["f1_2"]))
            em = np.where(search, data["em_3"], np.where(reason, reason_em, data["em_2"]))
            search_latency = data["latency_3"] + (
                data["reader_2_latency"] if mode == "reader_aware" else 0.0
            )
            latency = np.where(
                search, search_latency,
                data["latency_2"] + np.where(reason, reason_extra_latency, 0.0),
            )
            search_tokens = data["tokens_3"] + (
                data["tokens_2"] if mode == "reader_aware" else 0.0
            )
            tokens = np.where(
                search, search_tokens,
                data["tokens_2"] + np.where(reason, reason_extra_tokens, 0.0),
            )
            p_search = float(search.mean())
            p_reason = float(reason.mean())
            p_stop = 1.0 - p_search - p_reason
            random_expected = (
                p_search * float(data["f1_3"].mean())
                + p_reason * float(reason_f1.mean())
                + p_stop * float(data["f1_2"].mean())
            )
            random_difference = (
                f1 - p_search * data["f1_3"]
                - p_reason * reason_f1 - p_stop * data["f1_2"]
            )
            result = {
                "routes": {route: int(np.sum(routes == route)) for route in ("SEARCH", "REASON", "STOP")},
                "average_additional_searches": 2 + p_search,
                "average_reader_calls": 1 + (p_search if mode == "reader_aware" else 0) + 2 * p_reason,
                "em": float(em.mean() * 100),
                "f1": float(f1.mean() * 100),
                "estimated_latency_seconds": float(latency.mean()),
                "reader_generated_tokens": float(tokens.mean()),
                "vs_fixed3_f1": _compare(f1, data["f1_3"], seed=42),
                "vs_old_full_f1": _compare(f1, old_f1, seed=43),
                "matched_random_expected_f1": random_expected * 100,
                "vs_matched_random_f1": {
                    "change_points": float(random_difference.mean() * 100),
                    "ci95": _paired_interval(random_difference, 3000, 44),
                    "fraction_random_at_least_policy": _random_three_way_fraction(
                        data["f1_2"], data["f1_3"], reason_f1, routes,
                        float(f1.mean()), seed=45,
                    ),
                },
            }
            # Hold the original REASON queue and SEARCH count fixed.  This
            # isolates search targeting from spending more on reasoning.
            fixed_reason = old_routes == "REASON"
            old_search_count = int(np.sum(old_routes == "SEARCH"))
            eligible = np.flatnonzero(~fixed_reason)
            selected = eligible[np.argsort(score[eligible], kind="stable")[-old_search_count:]]
            isolated_search = np.zeros(len(ids), dtype=bool)
            isolated_search[selected] = True
            isolated_f1 = np.where(
                isolated_search, data["f1_3"],
                np.where(fixed_reason, reason_f1, data["f1_2"]),
            )
            search_gain = data["f1_3"] - data["f1_2"]
            selection_rate = old_search_count / len(eligible)
            isolated_minus_random = (
                isolated_search.astype(float) * search_gain
                - selection_rate * (~fixed_reason).astype(float) * search_gain
            )
            rng = np.random.default_rng(46)
            observed_search_gain = float(search_gain[isolated_search].sum())
            random_at_least = 0
            for _ in range(5000):
                random_selected = rng.choice(eligible, size=old_search_count, replace=False)
                random_at_least += float(search_gain[random_selected].sum()) >= observed_search_gain - 1e-12
            result["search_targeting_with_old_reason_queue_and_search_count"] = {
                "search_count": old_search_count,
                "reason_count": int(fixed_reason.sum()),
                "f1": float(isolated_f1.mean() * 100),
                "vs_old_full_f1": _compare(isolated_f1, old_f1, seed=47),
                "vs_matched_random_f1": {
                    "change_points": float(isolated_minus_random.mean() * 100),
                    "ci95": _paired_interval(isolated_minus_random, 3000, 48),
                    "fraction_random_at_least_policy": random_at_least / 5000,
                },
            }
            item["variants"][mode] = result
            print(
                f"{dataset}/{mode}: F1 {result['f1']:.3f}; "
                f"vs old full {result['vs_old_full_f1']['change_points']:+.3f}; "
                f"vs matched random {result['vs_matched_random_f1']['change_points']:+.3f}; "
                f"routes={result['routes']}",
                flush=True,
            )
        report["datasets"][dataset] = item
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
