"""Gate existing REASON routes by predicted *actual* answer-F1 benefit.

The full-validation REASON queue trains the gate; only the disjoint 500-item
per-dataset train holdout is evaluated.  SEARCH/STOP assignments remain frozen.
This is a cached-counterfactual, CPU-only experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from experiments.evaluate_answer_gain_search_policy import _paired_interval
from experiments.train_reader_free_student_policy import policy_feature_names, policy_features


ROOT = Path("data/benchmarks/multihop/context_reader_aware")
TRAIN_ROOT = ROOT / "full_validation"
HOLDOUT_ROOT = ROOT / "disjoint_holdout_seed20260916_500"
HOLDOUT_ANALYSIS = ROOT / "analysis" / "disjoint_holdout_seed20260916_500"
OUTPUT = ROOT / "analysis" / "reason_gain_gate_holdout.json"
MODEL_OUT = Path("data/models/reason_gain_gate.joblib")
DATASETS = ("hotpotqa", "musique", "2wiki")


def _jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def _predictions(path: Path) -> dict[str, dict]:
    rows = list(_jsonl(path))
    by_id = {str(row["id"]): row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"Duplicate prediction IDs in {path}")
    return by_id


def _judge_rows(roots: list[Path], dataset: str, split: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for root in roots:
        path = root / f"{dataset}_{split}.jsonl"
        if not path.exists():
            continue
        for row in _jsonl(path):
            identifier = str(row["id"])
            if identifier in result:
                raise ValueError(f"Duplicate judge ID: {identifier}")
            result[identifier] = row
    return result


def _selected_features(
    prefix_root: Path, dataset: str, split: str, names: tuple[str, ...],
    predictions: dict[str, dict], judged: dict[str, dict],
) -> dict:
    identifiers, features, gain, baseline_f1, baseline_em = [], [], [], [], []
    new_f1, new_em, reason_latency, reason_tokens = [], [], [], []
    selected_ids = {identifier for identifier, prediction in predictions.items()
                    if prediction["dataset"] == dataset and prediction["route"] == "REASON"}
    if selected_ids != set(judged) and split == "validation":
        raise ValueError(f"Validation REASON queue/judge coverage differs: {dataset}")
    path = prefix_root / f"{dataset}_{split}.jsonl"
    for row in _jsonl(path):
        identifier = str(row["id"])
        if identifier not in selected_ids:
            continue
        judge = judged[identifier]
        old = predictions[identifier]
        if abs(float(old["f1"]) - float(judge["judged_f1"])) > 1e-12:
            raise ValueError(f"Old answer and judged counterfactual differ: {identifier}")
        identifiers.append(identifier)
        features.append(policy_features(row, names, "single_reader"))
        baseline_f1.append(float(judge["baseline_f1"]))
        baseline_em.append(float(judge["baseline_em"]))
        new_f1.append(float(judge["judged_f1"]))
        new_em.append(float(judge["judged_em"]))
        gain.append(float(judge["judged_f1"]) - float(judge["baseline_f1"]))
        reason_latency.append(
            float(judge["repair_latency_seconds"]) + float(judge["judge_latency_seconds"])
        )
        reason_tokens.append(
            float(judge["repair_generated_tokens"]) + float(judge["judge_generated_tokens"])
        )
    if set(identifiers) != selected_ids:
        raise ValueError(f"Missing selected REASON prefixes: {dataset}")
    return {
        "ids": identifiers,
        "x": np.vstack(features),
        "gain": np.asarray(gain),
        "baseline_f1": np.asarray(baseline_f1),
        "baseline_em": np.asarray(baseline_em),
        "judged_f1": np.asarray(new_f1),
        "judged_em": np.asarray(new_em),
        "reason_latency": np.asarray(reason_latency),
        "reason_tokens": np.asarray(reason_tokens),
    }


def _model(kind: str, seed: int):
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=3.0, random_state=seed,
        )
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=180, max_features=0.7, min_samples_leaf=5,
            n_jobs=1, random_state=seed,
        )
    raise ValueError(kind)


def _oof(x: np.ndarray, y: np.ndarray, strata: np.ndarray, kind: str) -> np.ndarray:
    result = np.empty(len(y))
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=73)
    for fold, (fit, test) in enumerate(folds.split(x, strata)):
        model = _model(kind, 73 + fold)
        model.fit(x[fit], y[fit])
        result[test] = model.predict(x[test])
    return result


def _summary(
    score: np.ndarray, data: dict, old: dict[str, dict], keep: np.ndarray,
    *, random_draws: int = 5000,
) -> dict:
    n = len(data["ids"])
    old_f1 = np.asarray([float(old[identifier]["f1"]) for identifier in data["ids"]])
    old_em = np.asarray([float(old[identifier]["em"]) for identifier in data["ids"]])
    selected_f1 = np.where(keep, data["judged_f1"], data["baseline_f1"])
    selected_em = np.where(keep, data["judged_em"], data["baseline_em"])
    lost_gain = selected_f1 - old_f1
    keep_rate = float(keep.mean())
    matched_random_expected = float(data["baseline_f1"].mean() + keep_rate * data["gain"].mean())
    random_difference = keep.astype(float) * data["gain"] - keep_rate * data["gain"]
    rng = np.random.default_rng(74)
    observed = float(data["gain"][keep].sum())
    at_least = 0
    for _ in range(random_draws):
        chosen = rng.choice(n, size=int(keep.sum()), replace=False)
        at_least += float(data["gain"][chosen].sum()) >= observed - 1e-12
    return {
        "reason_kept": int(keep.sum()),
        "reason_skipped": int((~keep).sum()),
        "mean_predicted_gain_points": float(score.mean() * 100),
        "mean_actual_gain_kept_points": float(data["gain"][keep].mean() * 100) if keep.any() else None,
        "queue_f1": float(selected_f1.mean() * 100),
        "queue_em": float(selected_em.mean() * 100),
        "queue_f1_change_vs_old_points": float(lost_gain.mean() * 100),
        "queue_f1_change_vs_old_ci95": _paired_interval(lost_gain, 3000, 75),
        "matched_random_expected_queue_f1": matched_random_expected * 100,
        "queue_f1_change_vs_matched_random_points": float(random_difference.mean() * 100),
        "queue_f1_change_vs_matched_random_ci95": _paired_interval(random_difference, 3000, 76),
        "fraction_random_at_least_policy": at_least / random_draws,
        "saved_latency_seconds_per_queue_question": float(
            np.where(keep, 0.0, data["reason_latency"]).mean()
        ),
        "saved_reader_generated_tokens_per_queue_question": float(
            np.where(keep, 0.0, data["reason_tokens"]).mean()
        ),
    }


def main() -> None:
    full_report = json.loads((ROOT / "analysis" / "dual_value_full_system.json").read_text())
    judge_roots = [Path(path) for path in full_report["judge_roots"]]
    train_predictions = _predictions(ROOT / "analysis" / "dual_value_full_system.predictions.jsonl")
    holdout_predictions = _predictions(HOLDOUT_ANALYSIS / "full_system.predictions.jsonl")
    manifest = json.loads((TRAIN_ROOT / "hotpotqa_validation.manifest.json").read_text())
    names = tuple(manifest["feature_names"])
    train_parts = [
        _selected_features(
            TRAIN_ROOT, dataset, "validation", names, train_predictions,
            _judge_rows(judge_roots, dataset, "validation"),
        )
        for dataset in DATASETS
    ]
    train_x = np.vstack([part["x"] for part in train_parts])
    train_y = np.concatenate([part["gain"] for part in train_parts])
    strata = np.concatenate([
        np.asarray([f"{dataset}:{'positive' if value > 0 else 'negative' if value < 0 else 'zero'}"
                    for value in part["gain"]])
        for dataset, part in zip(DATASETS, train_parts, strict=True)
    ])
    oof_results = {}
    for kind in ("hgb", "extra_trees"):
        print(f"Cross-fitting {kind} on {len(train_y):,} REASON examples", flush=True)
        prediction = _oof(train_x, train_y, strata, kind)
        rank = np.argsort(prediction, kind="stable")
        keep = np.zeros(len(train_y), dtype=bool)
        keep[rank[-round(len(train_y) * 0.5):]] = True
        oof_results[kind] = {
            "mse": float(np.mean((prediction - train_y) ** 2)),
            "rank50_advantage_vs_random_points": float(
                (np.mean(keep * train_y) - 0.5 * np.mean(train_y)) * 100
            ),
            "positive_threshold_rate": float(np.mean(prediction > 0)),
            "positive_threshold_gain_points": float(np.mean((prediction > 0) * train_y) * 100),
        }
    chosen_kind = max(oof_results, key=lambda kind: (
        oof_results[kind]["rank50_advantage_vs_random_points"],
        -oof_results[kind]["mse"],
    ))
    print(f"Chosen on training OOF: {chosen_kind}", flush=True)
    model = _model(chosen_kind, 73)
    model.fit(train_x, train_y)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": policy_feature_names(names, "single_reader")}, MODEL_OUT)
    report = {
        "method": "predict_actual_judged_f1_minus_step2_f1_on_existing_reason_queue",
        "training_scope": "full-validation existing REASON queue only",
        "training_questions": len(train_y),
        "training_counts": {dataset: len(part["ids"]) for dataset, part in zip(DATASETS, train_parts, strict=True)},
        "training_label_counts": {
            "positive": int(np.sum(train_y > 0)),
            "negative": int(np.sum(train_y < 0)),
            "zero": int(np.sum(train_y == 0)),
        },
        "model_selection": "3-fold training-only OOF rank advantage at 50% REASON budget",
        "model": chosen_kind,
        "oof": oof_results,
        "holdout": {},
        "notes": [
            "SEARCH and original REASON-queue membership are held fixed.",
            "The gate is trained on only the old policy's REASON queue, so it is not validated for adding REASON to new questions.",
            "The same 500-question holdout was examined in prior exploratory work; this is not a pristine final test.",
            "All latencies are cached estimates, not live end-to-end measurements.",
            "No Qwen inference was run for this experiment.",
        ],
    }
    train_ids = {identifier for part in train_parts for identifier in part["ids"]}
    for dataset in DATASETS:
        judged = _judge_rows([HOLDOUT_ANALYSIS / "all_reason_judge"], dataset, "train")
        data = _selected_features(
            HOLDOUT_ROOT, dataset, "train", names, holdout_predictions, judged,
        )
        if train_ids.intersection(data["ids"]):
            raise ValueError(f"Training/evaluation ID overlap: {dataset}")
        score = model.predict(data["x"])
        rank = np.argsort(score, kind="stable")
        keep_half = np.zeros(len(score), dtype=bool)
        keep_half[rank[-round(len(score) * 0.5):]] = True
        old = {identifier: holdout_predictions[identifier] for identifier in data["ids"]}
        old_all_f1 = np.asarray([
            float(row["f1"]) for row in holdout_predictions.values() if row["dataset"] == dataset
        ])
        old_all_em = np.asarray([
            float(row["em"]) for row in holdout_predictions.values() if row["dataset"] == dataset
        ])
        old_all_latency = np.asarray([
            float(row["latency_seconds"]) for row in holdout_predictions.values() if row["dataset"] == dataset
        ])
        old_all_tokens = np.asarray([
            float(row["reader_generated_tokens"]) for row in holdout_predictions.values() if row["dataset"] == dataset
        ])
        n_all = len(old_all_f1)
        item = {
            "questions": n_all,
            "reason_queue_questions": len(data["ids"]),
            "old_full_f1": float(old_all_f1.mean() * 100),
            "old_full_em": float(old_all_em.mean() * 100),
            "old_full_estimated_latency_seconds": float(old_all_latency.mean()),
            "old_full_reader_generated_tokens": float(old_all_tokens.mean()),
            "policies": {},
        }
        for name, keep in (
            ("predicted_positive_gain", score > 0.0),
            ("predicted_gain_over_1_point", score > 0.01),
            ("ranked_keep_half", keep_half),
        ):
            result = _summary(score, data, old, keep)
            item["policies"][name] = {
                **result,
                "full_f1": float(old_all_f1.mean() * 100 + result["queue_f1_change_vs_old_points"] * len(keep) / n_all),
                "full_f1_change_vs_old_points": float(result["queue_f1_change_vs_old_points"] * len(keep) / n_all),
                "full_f1_change_vs_old_ci95": [
                    float(bound * len(keep) / n_all)
                    for bound in result["queue_f1_change_vs_old_ci95"]
                ],
                "full_f1_change_vs_matched_random_points": float(
                    result["queue_f1_change_vs_matched_random_points"] * len(keep) / n_all
                ),
                "full_f1_change_vs_matched_random_ci95": [
                    float(bound * len(keep) / n_all)
                    for bound in result["queue_f1_change_vs_matched_random_ci95"]
                ],
                "full_em": float(old_all_em.mean() * 100 + (result["queue_em"] - data["judged_em"].mean() * 100) * len(keep) / n_all),
                "full_estimated_latency_seconds": float(
                    old_all_latency.mean() - result["saved_latency_seconds_per_queue_question"] * len(keep) / n_all
                ),
                "full_reader_generated_tokens": float(
                    old_all_tokens.mean() - result["saved_reader_generated_tokens_per_queue_question"] * len(keep) / n_all
                ),
            }
            print(
                f"{dataset}/{name}: kept {result['reason_kept']}/{len(keep)} REASON, "
                f"full F1 {item['policies'][name]['full_f1']:.3f} "
                f"({result['queue_f1_change_vs_old_points'] * len(keep) / n_all:+.3f} vs old)",
                flush=True,
            )
        report["holdout"][dataset] = item
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
