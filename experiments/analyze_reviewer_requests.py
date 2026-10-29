"""Offline analyses requested in the KCI review, replayed on the published numerical data.

1. Search-cost (lambda) sweep: F1 against average searches.
2. Contribution of the four feature groups (validation permutation and
   drop-group retraining) and the individually most important features.
3. Oracle upper bound, premature-stop / over-search losses, and alternative
   training targets (one-step gain, quantile gain, P(gain > 0)).
4. Output-token and latency decomposition, and a reader-free policy that
   decides without generating intermediate answers.

No LLM is called. Inputs are the Release replay arrays under artifacts/replay/
and the five published policy models; every threshold is calibrated on
grouped train out-of-fold predictions exactly as in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from experiments.evaluate_answer_gain_search_policy import DATASETS, ROOT, _model
from experiments.evaluate_sequential_answer_gain_policy import _calibrate, _future_gain, _simulate

REPLAY = Path("artifacts/replay")
MODELS = Path("data/models/sequential_answer_gain_multiseed")
SEEDS = tuple(range(42, 47))
BASE_COST = 0.001
LAMBDAS = (0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
GROUPS = ("query", "retrieval", "evidence", "answer")
QUANTILES = (0.95,)  # 89% of gain targets are exactly 0, so q <= 0.9 predicts a constant 0
MARGINS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, float("inf"))
HGB = dict(learning_rate=0.05, max_iter=180, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=3.0)


class GainProbability(HistGradientBoostingClassifier):
    """Scores P(future gain > 0) so the same threshold calibration applies."""

    def fit(self, x, y):
        return super().fit(x, y > 0)

    def predict(self, x):
        return self.predict_proba(x)[:, 1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "reviewer_requests.json")
    parser.add_argument("--permutation-repeats", type=int, default=3)
    return parser.parse_args()


def _load(folder: str) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for dataset in DATASETS:
        with np.load(REPLAY / folder / f"{dataset}.npz", allow_pickle=False) as arrays:
            result[dataset] = {key: arrays[key] for key in arrays.files}
    return result


def _concat(data: dict) -> dict[str, np.ndarray]:
    keys = ("x", "f1", "em", "reader_tokens", "estimated_latency", "fixed4_latency",
            "fixed4_tokens", "early_reader_latency")
    merged = {key: np.concatenate([data[dataset][key] for dataset in DATASETS]) for key in keys}
    merged["strata"] = np.concatenate([np.full(len(data[d]["f1"]), d) for d in DATASETS])
    return merged


def feature_groups(names: tuple[str, ...]) -> dict[str, list[int]]:
    """Paper Section III-2: retrieval 12 and evidence 11 per step, query 15, answer 21."""
    groups = {group: [] for group in GROUPS}
    for index, name in enumerate(names):
        if name.startswith("question_"):
            groups["query"].append(index)
        elif name.startswith("current_"):
            groups["answer"].append(index)
        elif name.startswith("step"):
            groups["retrieval" if index % 23 < 12 else "evidence"].append(index)
        else:
            raise ValueError(f"Unknown feature group: {name}")
    if [len(groups[group]) for group in GROUPS] != [15, 36, 33, 21]:
        raise ValueError("Feature groups do not match the paper's 15/36/33/21 split")
    return groups


def _flat(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, x.shape[-1])


def _predict(model, x: np.ndarray) -> np.ndarray:
    return model.predict(_flat(x)).reshape(-1, 3)


def _oof(x: np.ndarray, target: np.ndarray, strata: np.ndarray, factory) -> np.ndarray:
    """Same grouped 3-fold split and seeds as the paper's calibration."""
    oof = np.empty(target.shape)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (fit, test) in enumerate(folds.split(np.zeros(len(x)), strata)):
        model = factory(42 + fold)
        model.fit(_flat(x[fit]), target[fit].reshape(-1))
        oof[test] = _predict(model, x[test])
    return oof


def _train_variant(train: dict, factory, target: np.ndarray, columns: list[int]):
    x = train["x"][:, :, columns]
    oof = _oof(x, target, train["strata"], factory)
    thresholds, _ = _calibrate(oof, train["f1"], BASE_COST)
    model = factory(42)
    model.fit(_flat(x), target.reshape(-1))
    return model, thresholds


def _intermediate_reader_seconds(data: dict, steps: np.ndarray) -> np.ndarray:
    """Reader time spent on answers before the stopping step.

    The replay stores cumulative latency, the first two reader calls together,
    and fixed-4 latency, so steps 2 and 3 are exact; step 1 splits the first
    two calls evenly (the reader context is capped at the same length).
    """
    early = data["early_reader_latency"]
    first_three = data["estimated_latency"][:, 3] - data["fixed4_latency"]
    options = np.column_stack([np.zeros_like(early), early / 2, early, first_three])
    return options[np.arange(len(steps)), steps]


def _metrics(data: dict, steps: np.ndarray, *, reader_free: bool = False) -> dict:
    indices = np.arange(len(steps))
    f1 = data["f1"]
    chosen = f1[indices, steps]
    best = f1.max(axis=1)
    best_step = f1.argmax(axis=1)
    loss = best - chosen
    per_step_tokens = np.diff(data["reader_tokens"], axis=1, prepend=0.0)
    if reader_free:
        tokens = per_step_tokens[indices, steps]
        latency = data["estimated_latency"][indices, steps] - _intermediate_reader_seconds(data, steps)
    else:
        tokens = data["reader_tokens"][indices, steps]
        latency = data["estimated_latency"][indices, steps]
    depth_rates = np.bincount(steps, minlength=4) / len(steps)
    future_best = np.column_stack([f1[:, s + 1:].max(axis=1) for s in range(3)] + [np.full(len(f1), -np.inf)])
    later_better = future_best[indices, steps] > chosen
    return {
        "f1": float(chosen.mean() * 100),
        "em": float(data["em"][indices, steps].mean() * 100),
        "average_total_searches": float(1 + steps.mean()),
        "reader_calls": float(1.0 if reader_free else 1 + steps.mean()),
        "reader_generated_tokens": float(tokens.mean()),
        "estimated_latency_seconds": float(latency.mean()),
        "f1_change_vs_fixed4_points": float((chosen - f1[:, 3]).mean() * 100),
        "f1_change_vs_matched_depth_random_points": float((chosen - f1 @ depth_rates).mean() * 100),
        "gap_to_oracle_points": float(loss.mean() * 100),
        "premature_stop_rate": float(later_better.mean() * 100),
        "premature_stop_loss_points": float(np.where(best_step > steps, loss, 0).mean() * 100),
        "over_search_loss_points": float(np.where(best_step < steps, loss, 0).mean() * 100),
        "search_distribution": [float(rate * 100) for rate in depth_rates],
    }


def _hybrid(data: dict, full_pred: np.ndarray, full_thresholds: np.ndarray,
            free_pred: np.ndarray, free_thresholds: np.ndarray, margin: float) -> dict:
    """Reader-free gate first; generate the current answer only near its threshold.

    An answer is generated at an uncertain step (then the full policy decides)
    and at the stopping step (the final answer). margin=0 is the reader-free
    policy and margin=inf the published policy.
    """
    n = len(full_pred)
    steps = np.full(n, 3)
    calls = np.zeros((n, 4), dtype=bool)
    active = np.ones(n, dtype=bool)
    for step in range(3):
        distance = free_pred[:, step] - free_thresholds[step]
        unsure = active & (np.abs(distance) <= margin)
        calls[unsure, step] = True
        stop = active & np.where(unsure, full_pred[:, step] <= full_thresholds[step], distance <= 0)
        steps[stop] = step
        active &= ~stop
    indices = np.arange(n)
    calls[indices, steps] = True
    early = data["early_reader_latency"]
    reader_seconds = np.column_stack([early / 2, early / 2,
                                      data["estimated_latency"][:, 3] - data["fixed4_latency"] - early,
                                      np.zeros(n)])
    skipped = (~calls) & (np.arange(4) < steps[:, None])
    result = _metrics(data, steps)
    result["reader_calls"] = float(calls.sum(axis=1).mean())
    result["reader_generated_tokens"] = float(
        (np.diff(data["reader_tokens"], axis=1, prepend=0.0) * calls).sum(axis=1).mean())
    result["estimated_latency_seconds"] = float(
        (data["estimated_latency"][indices, steps] - (reader_seconds * skipped).sum(axis=1)).mean())
    return result


def _per_dataset(validation: dict, model, thresholds: np.ndarray, columns, **kwargs) -> dict:
    result = {}
    for dataset in DATASETS:
        data = validation[dataset]
        steps = _simulate(_predict(model, data["x"][:, :, columns]), thresholds)
        result[dataset] = _metrics(data, steps, **kwargs)
    return result


def _reference(validation: dict) -> dict:
    result = {}
    for dataset in DATASETS:
        data = validation[dataset]
        n = len(data["f1"])
        oracle = data["f1"].argmax(axis=1)
        result[dataset] = {
            "fixed": {str(k + 1): _metrics(data, np.full(n, k), reader_free=True) for k in range(4)},
            "oracle_not_deployable": _metrics(data, oracle),
            "fixed4_single_reader": {
                "reader_generated_tokens": float(data["fixed4_tokens"].mean()),
                "estimated_latency_seconds": float(data["fixed4_latency"].mean()),
            },
            "per_step_reader_tokens": np.diff(data["reader_tokens"], axis=1, prepend=0.0).mean(0).tolist(),
            "estimated_latency_by_depth": data["estimated_latency"].mean(0).tolist(),
        }
    return result


def _permutation(validation: dict, model, thresholds, columns: list[int], repeats: int, seed: int) -> dict:
    """Shuffle whole question rows of the given columns, keeping step structure."""
    result = {}
    for dataset in DATASETS:
        data = validation[dataset]
        target = _future_gain(data["f1"])
        base_pred = _predict(model, data["x"])
        base_mse = float(np.mean((base_pred - target) ** 2))
        base_f1 = data["f1"][np.arange(len(target)), _simulate(base_pred, thresholds)].mean()
        mse, f1, searches = [], [], []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed * 1000 + repeat)
            x = data["x"].copy()
            order = rng.permutation(len(x))
            x[:, :, columns] = data["x"][order][:, :, columns]
            pred = _predict(model, x)
            steps = _simulate(pred, thresholds)
            mse.append(np.mean((pred - target) ** 2))
            f1.append(data["f1"][np.arange(len(steps)), steps].mean())
            searches.append(1 + steps.mean())
        result[dataset] = {
            "mse_increase_percent": float((np.mean(mse) / base_mse - 1) * 100),
            "f1_change_points": float((np.mean(f1) - base_f1) * 100),
            "average_total_searches": float(np.mean(searches)),
        }
    return result


def _feature_permutation(validation: dict, model, repeats: int, seed: int) -> np.ndarray:
    """Per-feature MSE increase (%) on the pooled validation states."""
    x = np.concatenate([validation[d]["x"] for d in DATASETS])
    target = np.concatenate([_future_gain(validation[d]["f1"]) for d in DATASETS])
    base = np.mean((_predict(model, x) - target) ** 2)
    rng = np.random.default_rng(seed)
    increase = np.zeros(x.shape[-1])
    for column in range(x.shape[-1]):
        values = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[:, :, column] = x[rng.permutation(len(x)), :, column]
            values.append(np.mean((_predict(model, shuffled) - target) ** 2))
        increase[column] = (np.mean(values) / base - 1) * 100
    return increase


def _aggregate(per_seed: list[dict]) -> dict:
    """Mean and sample std across seeds for every numeric leaf."""
    first = per_seed[0]
    if isinstance(first, dict):
        return {key: _aggregate([item[key] for item in per_seed]) for key in first}
    if isinstance(first, list):
        values = np.asarray(per_seed, dtype=float)
        return {"mean": values.mean(0).tolist(), "sample_std": values.std(0, ddof=1).tolist()}
    values = np.asarray(per_seed, dtype=float)
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}


def main() -> None:
    args = _arguments()
    validation = _load("full_validation")
    names = tuple(validation[DATASETS[0]]["feature_names"])
    groups = feature_groups(names)
    all_columns = list(range(len(names)))
    hgb = lambda seed: _model("hgb", seed)  # noqa: E731
    variants = {
        "one_step_gain_target": hgb,
        **{f"quantile_{q}": (lambda q: lambda seed: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, random_state=seed, **HGB))(q) for q in QUANTILES},
        "gain_positive_probability": lambda seed: GainProbability(random_state=seed, **HGB),
    }
    seeds = {}
    for seed in SEEDS:
        print(f"seed {seed}", flush=True)
        saved = joblib.load(MODELS / f"policy_seed{seed}.joblib")
        if tuple(saved["feature_names"]) != names:
            raise ValueError("Feature schema mismatch")
        model, published = saved["model"], np.asarray(saved["thresholds"])
        train = _concat(_load(f"train_pool_seed{seed}_2000"))
        target = _future_gain(train["f1"])
        oof = _oof(train["x"], target, train["strata"], hgb)
        recalibrated, _ = _calibrate(oof, train["f1"], BASE_COST)
        row = {
            "published_thresholds": published.tolist(),
            "recalibrated_thresholds": recalibrated.tolist(),
            "policy": _per_dataset(validation, model, published, all_columns),
            "lambda_sweep": {},
            "group_permutation": {},
            "drop_group": {},
            "targets": {},
        }
        for cost in LAMBDAS:
            thresholds, _ = _calibrate(oof, train["f1"], cost)
            row["lambda_sweep"][str(cost)] = {
                "thresholds": thresholds.tolist(),
                **_per_dataset(validation, model, thresholds, all_columns),
            }
        for group in GROUPS:
            row["group_permutation"][group] = _permutation(
                validation, model, published, groups[group], args.permutation_repeats, seed)
            kept = [c for c in all_columns if c not in groups[group]]
            variant, thresholds = _train_variant(train, hgb, target, kept)
            row["drop_group"][group] = _per_dataset(validation, variant, thresholds, kept,
                                                    reader_free=group == "answer")
            if group == "answer":
                row["hybrid_reader_gate"] = {
                    str(margin): {
                        dataset: _hybrid(validation[dataset], _predict(model, validation[dataset]["x"]),
                                         published, _predict(variant, validation[dataset]["x"][:, :, kept]),
                                         thresholds, margin)
                        for dataset in DATASETS
                    }
                    for margin in MARGINS
                }
        one_step = np.column_stack([train["f1"][:, s + 1] - train["f1"][:, s] for s in range(3)])
        for name, factory in variants.items():
            variant, thresholds = _train_variant(
                train, factory, one_step if name == "one_step_gain_target" else target, all_columns)
            row["targets"][name] = _per_dataset(validation, variant, thresholds, all_columns)
        row["feature_permutation_mse_increase_percent"] = _feature_permutation(
            validation, model, args.permutation_repeats, seed).tolist()
        seeds[str(seed)] = row
    report = {
        "method": "reviewer_requested_offline_analyses",
        "replay_root": str(REPLAY),
        "search_cost_for_variants": BASE_COST,
        "lambdas": list(LAMBDAS),
        "feature_names": list(names),
        "feature_groups": groups,
        "reference": _reference(validation),
        "seeds": seeds,
        "across_seeds": _aggregate([
            {key: value for key, value in row.items() if not key.endswith("thresholds")}
            for row in seeds.values()
        ]),
        "notes": [
            "Offline replay of cached trajectories; no reader or planner call is made.",
            "Thresholds are calibrated on grouped train OOF predictions with the paper's procedure.",
            "Drop-group 'answer' is reader-free: one reader call at the stopping step only.",
            "Reader-free step-1 latency splits the stored first-two-call reader time evenly.",
            "Validation was used during earlier development; it is not an untouched test set.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
