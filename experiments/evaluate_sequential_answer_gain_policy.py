"""Train and evaluate a gold-free, every-prefix STOP/SEARCH policy.

The same gradient-boosted regressor observes the initial retrieval and each
subsequent prefix. Its training target is the best *future* answer-F1 gain,
not the gold hop count or a fixed search depth. Evaluation replays cached
prefixes; it does not run the frozen reader or measure live latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.analyze_student_policy_multiseed import _planner_elapsed
from experiments.evaluate_answer_gain_search_policy import DATASETS, ROOT, _model, _paired_interval
from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_reader_free_student_policy import policy_feature_names
from src.answer_gain_policy import sequential_state


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=ROOT / "train_pool_seed42_2000")
    parser.add_argument("--validation-root", type=Path, default=ROOT / "full_validation")
    parser.add_argument("--holdout-root", type=Path, default=ROOT / "disjoint_holdout_seed20260916_500")
    parser.add_argument("--skip-holdout", action="store_true",
                        help="Evaluate only full validation; later train seeds may overlap the old train-derived holdout")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "sequential_answer_gain_hgb.json")
    parser.add_argument("--model-out", type=Path, default=Path("data/models/sequential_answer_gain_hgb.joblib"))
    parser.add_argument("--search-cost", type=float, default=0.001,
                        help="F1 fraction charged per additional search during train-only calibration")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.search_cost < 0 or args.bootstrap_samples < 1:
        parser.error("search cost는 0 이상, bootstrap samples는 1 이상이어야 합니다.")
    return args


def _feature_names(manifest_names: tuple[str, ...]) -> tuple[str, ...]:
    names = policy_feature_names(manifest_names, "single_reader")
    base_count = len(policy_feature_names(manifest_names, "retrieval_only"))
    return names[:base_count] + tuple(
        name.replace("step2_reader_", "current_reader_").replace("step2_", "current_")
        for name in names[base_count:]
    )


def _features_by_step(row: dict, manifest_names: tuple[str, ...]) -> np.ndarray:
    """Use only prefixes available at the decision step; never future or gold data."""
    question = str(row.get("question", ""))
    actions = [row["actions"][name] for name in ACTIONS]
    for action in actions:
        if len(action["policy_features"]) != len(manifest_names):
            raise ValueError(f"{row.get('id')}: policy feature schema mismatch")
    states = np.vstack([
        sequential_state(question, actions, step, manifest_names)
        for step in range(len(ACTIONS) - 1)
    ])
    if states.shape[1] != len(_feature_names(manifest_names)):
        raise RuntimeError("Sequential feature schema mismatch")
    return states


def _read(root: Path, split: str, manifest_names: tuple[str, ...]) -> dict[str, dict]:
    result = {}
    for dataset in DATASETS:
        manifest_path = root / f"{dataset}_{split}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "ready" or tuple(manifest["feature_names"]) != manifest_names:
            raise ValueError(f"Incomplete or incompatible prefix data: {manifest_path}")
        ids, states, f1, em, estimated_latency, reader_tokens = [], [], [], [], [], []
        fixed4_latency, fixed4_tokens, early_reader_latency, early_reader_tokens = [], [], [], []
        with (root / f"{dataset}_{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                ids.append(str(row["id"]))
                states.append(_features_by_step(row, manifest_names))
                actions = [row["actions"][name] for name in ACTIONS]
                f1.append([float(action["f1"]) for action in actions])
                em.append([float(action["em"]) for action in actions])
                reader_seconds = np.asarray([
                    float(action.get("reader_latency_seconds") or 0.0) for action in actions
                ])
                reader_generated = np.asarray([
                    float(action.get("reader_generated_tokens") or 0.0) for action in actions
                ])
                initial = float(row.get("initial_retrieval_latency_seconds", 0.0))
                estimated_latency.append([
                    initial + _planner_elapsed(row, step) + float(reader_seconds[:step + 1].sum())
                    for step in range(len(ACTIONS))
                ])
                reader_tokens.append(np.cumsum(reader_generated))
                fixed4_latency.append(
                    initial + _planner_elapsed(row, len(ACTIONS) - 1) + reader_seconds[-1]
                )
                fixed4_tokens.append(reader_generated[-1])
                early_reader_latency.append(float(reader_seconds[:2].sum()))
                early_reader_tokens.append(float(reader_generated[:2].sum()))
        if len(ids) != int(manifest["completed_rows"]) or len(ids) != len(set(ids)):
            raise ValueError(f"Wrong number of unique rows: {dataset}/{split}")
        result[dataset] = {
            "ids": ids,
            "x": np.asarray(states, dtype="float64"),
            "f1": np.asarray(f1, dtype="float64"),
            "em": np.asarray(em, dtype="float64"),
            "estimated_latency": np.asarray(estimated_latency, dtype="float64"),
            "reader_tokens": np.asarray(reader_tokens, dtype="float64"),
            "fixed4_latency": np.asarray(fixed4_latency, dtype="float64"),
            "fixed4_tokens": np.asarray(fixed4_tokens, dtype="float64"),
            "early_reader_latency": np.asarray(early_reader_latency, dtype="float64"),
            "early_reader_tokens": np.asarray(early_reader_tokens, dtype="float64"),
        }
        print(f"Loaded {dataset}/{split}: {len(ids):,} questions", flush=True)
    return result


def _future_gain(f1: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.max(f1[:, step + 1:], axis=1) - f1[:, step]
        for step in range(len(ACTIONS) - 1)
    ])


def _simulate(predictions: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    steps = np.full(len(predictions), len(ACTIONS) - 1, dtype="int64")
    active = np.ones(len(predictions), dtype=bool)
    for step in range(len(ACTIONS) - 1):
        stop = active & (predictions[:, step] <= thresholds[step])
        steps[stop] = step
        active &= ~stop
    return steps


def _calibrate(predictions: np.ndarray, f1: np.ndarray, search_cost: float) -> tuple[np.ndarray, dict]:
    """Calibrate all decision thresholds on grouped out-of-fold train predictions."""
    n = len(f1)
    choices = [
        np.unique(np.r_[np.quantile(predictions[:, step], np.linspace(0, 1, 41)),
                        0.0, search_cost, -np.inf, np.inf])
        for step in range(len(ACTIONS) - 1)
    ]
    starts = [np.zeros(3), np.asarray([np.median(predictions[:, step]) for step in range(3)]),
              np.full(3, -np.inf)]
    best = None
    for start in starts:
        thresholds = start.copy()
        for _ in range(4):
            changed = False
            for step in range(3):
                current = None
                for candidate in choices[step]:
                    trial = thresholds.copy()
                    trial[step] = candidate
                    selected = _simulate(predictions, trial)
                    utility = float(np.mean(f1[np.arange(n), selected] - search_cost * selected))
                    key = (utility, -float(selected.mean()))
                    if current is None or key > current[0]:
                        current = (key, candidate)
                assert current is not None
                changed |= thresholds[step] != current[1]
                thresholds[step] = current[1]
            if not changed:
                break
        selected = _simulate(predictions, thresholds)
        utility = float(np.mean(f1[np.arange(n), selected] - search_cost * selected))
        key = (utility, -float(selected.mean()))
        if best is None or key > best[0]:
            best = (key, thresholds.copy(), selected)
    assert best is not None
    return best[1], {
        "f1": float(np.mean(f1[np.arange(n), best[2]]) * 100),
        "average_total_searches": float(1 + best[2].mean()),
        "utility_points": float(best[0][0] * 100),
    }


def _calibrate_late(predictions: np.ndarray, f1: np.ndarray, search_cost: float) -> float:
    """Train-only late-gate calibration for a fair same-model timing ablation."""
    candidates = np.unique(np.r_[np.quantile(predictions[:, 2], np.linspace(0, 1, 101)),
                                 0.0, search_cost, -np.inf, np.inf])
    indices = np.arange(len(f1))
    best = None
    for threshold in candidates:
        steps = np.where(predictions[:, 2] > threshold, 3, 2)
        objective = float(np.mean(f1[indices, steps] - search_cost * steps))
        key = (objective, -float(steps.mean()))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return best[1]


def _summary(data: dict, predictions: np.ndarray, thresholds: np.ndarray,
             late_threshold: float,
             bootstrap_samples: int) -> dict:
    steps = _simulate(predictions, thresholds)
    indices = np.arange(len(steps))
    f1 = data["f1"][indices, steps]
    em = data["em"][indices, steps]
    fixed4_f1 = data["f1"][:, -1]
    depth_rates = np.bincount(steps, minlength=len(ACTIONS)) / len(steps)
    matched_random_f1 = data["f1"] @ depth_rates
    oracle_steps = np.argmax(data["f1"], axis=1)
    late_steps = np.where(predictions[:, 2] > late_threshold, 3, 2)
    late_f1 = data["f1"][indices, late_steps]
    late_em = data["em"][indices, late_steps]
    result = {
        "questions": len(steps),
        "fixed": {
            str(step + 1): {
                "f1": float(data["f1"][:, step].mean() * 100),
                "em": float(data["em"][:, step].mean() * 100),
            }
            for step in range(len(ACTIONS))
        },
        "policy": {
            "f1": float(f1.mean() * 100),
            "em": float(em.mean() * 100),
            "average_total_searches": float(1 + steps.mean()),
            "average_reader_calls": float(1 + steps.mean()),
            "estimated_latency_seconds": float(data["estimated_latency"][indices, steps].mean()),
            "reader_generated_tokens": float(data["reader_tokens"][indices, steps].mean()),
            "search_distribution": {
                str(step + 1): int(np.sum(steps == step)) for step in range(len(ACTIONS))
            },
        },
        "fixed4_single_reader": {
            "estimated_latency_seconds": float(data["fixed4_latency"].mean()),
            "reader_generated_tokens": float(data["fixed4_tokens"].mean()),
        },
        "same_model_late_only_ablation": {
            "f1": float(late_f1.mean() * 100),
            "em": float(late_em.mean() * 100),
            "average_total_searches": float(1 + late_steps.mean()),
            "average_reader_calls": float(1 + np.mean(late_steps == 3)),
            "estimated_latency_seconds": float(
                (data["estimated_latency"][indices, late_steps]
                 - data["early_reader_latency"]).mean()
            ),
            "reader_generated_tokens": float(
                (data["reader_tokens"][indices, late_steps]
                 - data["early_reader_tokens"]).mean()
            ),
        },
        "f1_change_vs_same_model_late_only_points": float((f1 - late_f1).mean() * 100),
        "f1_change_vs_same_model_late_only_ci95": _paired_interval(
            f1 - late_f1, bootstrap_samples, 44
        ),
        "matched_depth_random_expected_f1": float(matched_random_f1.mean() * 100),
        "f1_change_vs_matched_depth_random_points": float(
            (f1 - matched_random_f1).mean() * 100
        ),
        "f1_change_vs_matched_depth_random_ci95": _paired_interval(
            f1 - matched_random_f1, bootstrap_samples, 43
        ),
        "oracle_diagnostic_not_deployable": {
            "best_prefix_f1": float(data["f1"][indices, oracle_steps].mean() * 100),
            "best_prefix_search_distribution": {
                str(step + 1): int(np.sum(oracle_steps == step))
                for step in range(len(ACTIONS))
            },
        },
        "f1_change_vs_fixed4_points": float((f1 - fixed4_f1).mean() * 100),
        "f1_change_vs_fixed4_ci95": _paired_interval(
            f1 - fixed4_f1, bootstrap_samples, 42
        ),
    }
    return result


def main() -> None:
    args = _arguments()
    manifests = {
        dataset: json.loads((args.train_root / f"{dataset}_train.manifest.json").read_text(encoding="utf-8"))
        for dataset in DATASETS
    }
    training_seeds = {manifest.get("query_seed") for manifest in manifests.values()}
    if len(training_seeds) != 1:
        raise ValueError(f"Mixed training sample seeds: {training_seeds}")
    training_seed = training_seeds.pop()
    names = tuple(manifests[DATASETS[0]]["feature_names"])
    training = _read(args.train_root, "train", names)
    x = np.concatenate([training[dataset]["x"] for dataset in DATASETS])
    f1 = np.concatenate([training[dataset]["f1"] for dataset in DATASETS])
    ids = [identifier for dataset in DATASETS for identifier in training[dataset]["ids"]]
    strata = np.asarray([dataset for dataset in DATASETS for _ in training[dataset]["ids"]])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate training question IDs")
    y = _future_gain(f1)
    oof = np.empty_like(y)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (fit, test) in enumerate(folds.split(np.zeros(len(ids)), strata)):
        model = _model("hgb", 42 + fold)
        model.fit(x[fit].reshape(-1, x.shape[-1]), y[fit].reshape(-1))
        oof[test] = model.predict(x[test].reshape(-1, x.shape[-1])).reshape(-1, 3)
        print(f"Grouped OOF fold {fold + 1}/3 complete", flush=True)
    thresholds, oof_summary = _calibrate(oof, f1, args.search_cost)
    late_threshold = _calibrate_late(oof, f1, args.search_cost)
    print(f"Train-OOF thresholds: {thresholds.tolist()} | {oof_summary}", flush=True)
    print(f"Train-OOF late-only threshold: {late_threshold}", flush=True)
    model = _model("hgb", 42)
    model.fit(x.reshape(-1, x.shape[-1]), y.reshape(-1))
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "thresholds": thresholds,
                 "late_only_threshold": late_threshold,
                 "feature_names": _feature_names(names), "manifest_feature_names": names,
                 "training_questions": len(ids), "training_states": int(y.size)}, args.model_out)
    report = {
        "method": "sequential_best_future_f1_gain_hgb",
        "model_file": str(args.model_out),
        "training_root": str(args.train_root),
        "training_sample_seed": training_seed,
        "training_cohort_by_dataset": {
            dataset: {
                "questions": len(training[dataset]["ids"]),
                "selected_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(training[dataset]["ids"])).encode("utf-8")
                ).hexdigest(),
            }
            for dataset in DATASETS
        },
        "validation_root": str(args.validation_root),
        "training_questions": len(ids),
        "training_states": int(y.size),
        "feature_count": x.shape[-1],
        "label": "best_future_f1_minus_current_f1",
        "search_cost_f1_fraction": args.search_cost,
        "thresholds": thresholds.tolist(),
        "late_only_threshold": late_threshold,
        "training_oof": oof_summary,
        "evaluations": {},
        "notes": [
            "Initial retrieval is mandatory; STOP/SEARCH is decided after each of the first three prefixes.",
            "At each decision the reader's current answer is observed, so reader calls equal total searches.",
            "Gold answer and future prefixes are used only for offline training labels/evaluation.",
            "Evaluation is cached-trajectory replay, not live end-to-end execution.",
            "Validation was used during earlier project development; it is not an untouched test set.",
        ],
    }
    train_ids = set(ids)
    cohorts = [("full_validation", args.validation_root, "validation")]
    if not args.skip_holdout:
        cohorts.append(("disjoint_holdout_500", args.holdout_root, "train"))
        report["notes"].append("The train-derived holdout has 500 questions per dataset.")
    for title, root, split in cohorts:
        cohort = _read(root, split, names)
        report["evaluations"][title] = {}
        for dataset in DATASETS:
            data = cohort[dataset]
            if train_ids.intersection(data["ids"]):
                raise ValueError(f"Train/evaluation ID overlap: {title}/{dataset}")
            predictions = model.predict(data["x"].reshape(-1, x.shape[-1])).reshape(-1, 3)
            result = _summary(data, predictions, thresholds, late_threshold, args.bootstrap_samples)
            report["evaluations"][title][dataset] = result
            print(f"{title}/{dataset}: F1={result['policy']['f1']:.3f}, "
                  f"fixed4={result['fixed']['4']['f1']:.3f}, "
                  f"searches={result['policy']['average_total_searches']:.3f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
