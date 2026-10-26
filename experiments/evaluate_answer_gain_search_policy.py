"""Test whether learning *answer* gain improves the third-search decision.

All thresholds/model choices are selected with cross-fitted training predictions.
The official validation and ID-disjoint train holdout are evaluation-only.  This
is a cached-trajectory counterfactual experiment, not a live latency benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from experiments.analyze_student_policy_multiseed import _planner_elapsed
from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_reader_free_student_policy import (
    policy_feature_names,
    policy_features,
    student_feature_names,
    student_features,
)


DATASETS = ("hotpotqa", "musique", "2wiki")
ROOT = Path("data/benchmarks/multihop/context_reader_aware")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=ROOT / "train_pool_seed42_2000")
    parser.add_argument("--validation-root", type=Path, default=ROOT / "full_validation")
    parser.add_argument(
        "--holdout-root", type=Path, default=ROOT / "disjoint_holdout_seed20260916_500"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "analysis" / "answer_gain_search_probe.json"
    )
    parser.add_argument(
        "--model-out", type=Path, default=Path("data/models/answer_gain_search_probe.joblib")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--random-draws", type=int, default=5000)
    parser.add_argument(
        "--feature-mode", choices=("retrieval_only", "single_reader"), default="retrieval_only"
    )
    return parser.parse_args()


def _model(kind: str, seed: int):
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=3.0,
            random_state=seed,
        )
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=180,
            max_features=0.7,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(kind)


def _read_dataset(
    root: Path, dataset: str, split: str, names: tuple[str, ...], feature_mode: str
):
    manifest = json.loads((root / f"{dataset}_{split}.manifest.json").read_text())
    if manifest.get("status") != "ready" or tuple(manifest["feature_names"]) != names:
        raise ValueError(f"Incomplete or incompatible prefix data: {root}/{dataset}_{split}")
    identifiers, features, f1_2, f1_3, em_2, em_3 = [], [], [], [], [], []
    latency_2, latency_3, reader_2_latency, tokens_2, tokens_3 = [], [], [], [], []
    with (root / f"{dataset}_{split}.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            identifiers.append(str(row["id"]))
            features.append(
                student_features(row, names)
                if feature_mode == "retrieval_only"
                else policy_features(row, names, "single_reader")
            )
            step2 = row["actions"][ACTIONS[2]]
            step3 = row["actions"][ACTIONS[3]]
            f1_2.append(float(step2["f1"]))
            f1_3.append(float(step3["f1"]))
            em_2.append(float(step2["em"]))
            em_3.append(float(step3["em"]))
            initial = float(row.get("initial_retrieval_latency_seconds", 0.0))
            reader2 = float(step2.get("reader_latency_seconds") or 0.0)
            latency_2.append(initial + _planner_elapsed(row, 2) + reader2)
            latency_3.append(
                initial + _planner_elapsed(row, 3)
                + float(step3.get("reader_latency_seconds") or 0.0)
            )
            reader_2_latency.append(reader2)
            tokens_2.append(float(step2.get("reader_generated_tokens") or 0.0))
            tokens_3.append(float(step3.get("reader_generated_tokens") or 0.0))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate question IDs: {dataset} {split}")
    if len(identifiers) != int(manifest["completed_rows"]):
        raise ValueError(f"Wrong number of prefix rows: {dataset} {split}")
    return {
        "ids": identifiers,
        "x": np.vstack(features),
        "f1_2": np.asarray(f1_2),
        "f1_3": np.asarray(f1_3),
        "em_2": np.asarray(em_2),
        "em_3": np.asarray(em_3),
        "latency_2": np.asarray(latency_2),
        "latency_3": np.asarray(latency_3),
        "reader_2_latency": np.asarray(reader_2_latency),
        "tokens_2": np.asarray(tokens_2),
        "tokens_3": np.asarray(tokens_3),
    }


def _paired_interval(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(samples)
    for start in range(0, samples, 100):
        end = min(samples, start + 100)
        indices = rng.integers(0, n, size=(end - start, n))
        means[start:end] = values[indices].mean(axis=1) * 100
    return [float(v) for v in np.quantile(means, (0.025, 0.975))]


def _random_fraction_at_least(gain: np.ndarray, mask: np.ndarray, draws: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    count = int(mask.sum())
    observed = float(gain[mask].sum())
    at_least = 0
    for _ in range(draws):
        chosen = rng.choice(len(gain), size=count, replace=False)
        at_least += float(gain[chosen].sum()) >= observed - 1e-12
    return at_least / draws


def _route_summary(
    data: dict, mask: np.ndarray, *, bootstrap: int, draws: int, seed: int,
    reader_aware: bool,
) -> dict:
    gain = data["f1_3"] - data["f1_2"]
    n = len(gain)
    rate = float(mask.mean())
    f1 = np.where(mask, data["f1_3"], data["f1_2"])
    em = np.where(mask, data["em_3"], data["em_2"])
    random_expected = float(data["f1_2"].mean() + rate * gain.mean())
    policy_minus_fixed3 = f1 - data["f1_3"]
    policy_minus_random = mask.astype(float) * gain - rate * gain
    search_latency = data["latency_3"] + (
        data["reader_2_latency"] if reader_aware else 0.0
    )
    search_tokens = data["tokens_3"] + (data["tokens_2"] if reader_aware else 0.0)
    return {
        "search_count": int(mask.sum()),
        "average_additional_searches": 2 + rate,
        "average_reader_calls": 1 + (rate if reader_aware else 0),
        "estimated_latency_seconds": float(np.where(mask, search_latency, data["latency_2"]).mean()),
        "reader_generated_tokens": float(np.where(mask, search_tokens, data["tokens_2"]).mean()),
        "em": float(em.mean() * 100),
        "f1": float(f1.mean() * 100),
        "f1_change_vs_fixed3_points": float(policy_minus_fixed3.mean() * 100),
        "f1_change_vs_fixed3_ci95": _paired_interval(policy_minus_fixed3, bootstrap, seed),
        "matched_random_expected_f1": random_expected * 100,
        "f1_change_vs_matched_random_points": float(policy_minus_random.mean() * 100),
        "f1_change_vs_matched_random_ci95": _paired_interval(
            policy_minus_random, bootstrap, seed + 1
        ),
        "fraction_random_at_least_policy": _random_fraction_at_least(
            gain, mask, draws, seed + 2
        ),
    }


def _oof_prediction(x: np.ndarray, y: np.ndarray, strata: np.ndarray, kind: str) -> np.ndarray:
    predictions = np.empty(len(y))
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(folds.split(x, strata)):
        model = _model(kind, 42 + fold)
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
    return predictions


def _threshold(oof: np.ndarray, gain: np.ndarray, search_cost: float) -> tuple[float, dict]:
    candidates = np.unique(
        np.concatenate((np.quantile(oof, np.linspace(0, 1, 101)), [0.0, -np.inf, np.inf]))
    )
    best = None
    for cutoff in candidates:
        chosen = oof > cutoff
        score = float(np.mean(chosen * (gain - search_cost)))
        item = (score, -float(chosen.mean()), float(cutoff))
        if best is None or item > best[0]:
            best = (item, {"threshold": float(cutoff), "search_rate": float(chosen.mean()),
                           "oof_policy_gain_points": float(np.mean(chosen * gain) * 100),
                           "oof_utility_points": score * 100})
    assert best is not None
    return best[1]["threshold"], best[1]


def _budget(root: Path, dataset: str) -> int:
    path = root.parent / "analysis" / "kci_required_baselines.json"
    if "disjoint_holdout" in root.name:
        path = root.parent / "analysis" / root.name / "kci_baselines_matched_three_way.json"
    report = json.loads(path.read_text())
    return int(report["datasets"][dataset]["matched_budget_random_pre_repair"]["search_count"])


def main() -> None:
    args = _arguments()
    manifest = json.loads((args.train_root / "hotpotqa_train.manifest.json").read_text())
    names = tuple(manifest["feature_names"])
    train_parts = [
        _read_dataset(args.train_root, dataset, "train", names, args.feature_mode)
        for dataset in DATASETS
    ]
    x = np.vstack([part["x"] for part in train_parts])
    y = np.concatenate([part["f1_3"] - part["f1_2"] for part in train_parts])
    strata = np.concatenate(
        [np.asarray([f"{dataset}:{'positive' if d > 0 else 'negative' if d < 0 else 'zero'}"
                     for d in part["f1_3"] - part["f1_2"]])
         for dataset, part in zip(DATASETS, train_parts, strict=True)]
    )
    train_ids = {identifier for part in train_parts for identifier in part["ids"]}
    oof_results = {}
    for kind in ("hgb", "extra_trees"):
        print(f"Cross-fitting {kind} on {len(y):,} training questions", flush=True)
        oof = _oof_prediction(x, y, strata, kind)
        order = np.argsort(oof, kind="stable")
        selected = np.zeros(len(y), dtype=bool)
        selected[order[-round(0.70 * len(y)):]] = True
        random_expected = float(selected.mean() * y.mean())
        threshold_pure, pure = _threshold(oof, y, 0.0)
        threshold_cost, costed = _threshold(oof, y, 0.001)
        oof_results[kind] = {
            "oof_mse": float(np.mean((oof - y) ** 2)),
            "rank70_gain_over_random_points": float((np.mean(selected * y) - random_expected) * 100),
            "pure_f1": pure,
            "cost_0_1_f1_point_per_search": costed,
        }
    chosen_kind = max(
        oof_results,
        key=lambda kind: (oof_results[kind]["rank70_gain_over_random_points"], -oof_results[kind]["oof_mse"]),
    )
    print(f"Chosen by training OOF rank@70%: {chosen_kind}", flush=True)
    model = _model(chosen_kind, 42)
    model.fit(x, y)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    feature_names = (
        student_feature_names(names)
        if args.feature_mode == "retrieval_only"
        else policy_feature_names(names, "single_reader")
    )
    joblib.dump({"model": model, "feature_names": feature_names,
                 "pure_threshold": oof_results[chosen_kind]["pure_f1"]["threshold"],
                 "cost_threshold": oof_results[chosen_kind]["cost_0_1_f1_point_per_search"]["threshold"]},
                args.model_out)
    report = {
        "method": "direct_answer_f1_gain_for_third_search",
        "feature_mode": args.feature_mode,
        "label": "f1_after_third_search_minus_f1_after_second_search",
        "training_questions": len(y),
        "training_question_ids_disjoint_from_evaluations": True,
        "features": x.shape[1],
        "model_selection": "3-fold training-only OOF rank gain at 70% search budget",
        "chosen_model": chosen_kind,
        "oof": oof_results,
        "model_file": str(args.model_out),
        "evaluation": {},
        "notes": [
            "No Qwen inference was run; answers and retrieval prefixes are cached.",
            "Matched-budget top-K ranking is a batch diagnostic, not an online stopping policy.",
            "All results here compare step-2 and step-3 answer trajectories; REASON is not evaluated.",
            "Official validation was used for earlier system development; the ID-disjoint holdout is only 500 per dataset.",
            "single_reader mode requires reading the step-2 answer before deciding SEARCH, adding a reader call on SEARCH paths.",
            "Latency is estimated from cached component timings, not measured end-to-end.",
        ],
    }
    for name, root, split in (
        ("full_validation", args.validation_root, "validation"),
        ("disjoint_holdout_500", args.holdout_root, "train"),
    ):
        report["evaluation"][name] = {}
        for dataset in DATASETS:
            data = _read_dataset(root, dataset, split, names, args.feature_mode)
            if train_ids.intersection(data["ids"]):
                raise ValueError(f"Train/evaluation question overlap: {name}/{dataset}")
            score = model.predict(data["x"])
            budget = _budget(root, dataset)
            order = np.argsort(score, kind="stable")
            ranked_mask = np.zeros(len(score), dtype=bool)
            ranked_mask[order[-budget:]] = True
            random_draws = args.random_draws if name == "disjoint_holdout_500" else min(1000, args.random_draws)
            result = {
                "count": len(score),
                "fixed2_em": float(data["em_2"].mean() * 100),
                "fixed2_f1": float(data["f1_2"].mean() * 100),
                "fixed3_em": float(data["em_3"].mean() * 100),
                "fixed3_f1": float(data["f1_3"].mean() * 100),
                "fixed3_estimated_latency_seconds": float(data["latency_3"].mean()),
                "fixed3_reader_generated_tokens": float(data["tokens_3"].mean()),
                "rank_at_existing_router_search_budget": _route_summary(
                    data, ranked_mask, bootstrap=args.bootstrap_samples,
                    draws=random_draws, seed=42,
                    reader_aware=args.feature_mode == "single_reader",
                ),
            }
            for key, objective in (("pure_f1", "pure_f1"),
                                   ("cost_0_1_f1_point", "cost_0_1_f1_point_per_search")):
                threshold = oof_results[chosen_kind][objective]["threshold"]
                result[key] = _route_summary(
                    data, score > threshold, bootstrap=args.bootstrap_samples,
                    draws=random_draws, seed=43 if key == "pure_f1" else 44,
                    reader_aware=args.feature_mode == "single_reader",
                )
            report["evaluation"][name][dataset] = result
            primary = result["pure_f1"]
            ranked = result["rank_at_existing_router_search_budget"]
            print(
                f"{name}/{dataset}: fixed3 F1={result['fixed3_f1']:.3f}, "
                f"pure F1={primary['f1']:.3f} ({primary['average_additional_searches']:.3f} searches), "
                f"matched-budget F1={ranked['f1']:.3f}, "
                f"ranked-minus-random={ranked['f1_change_vs_matched_random_points']:+.3f}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
