"""Train a stable tree ensemble for rare next-search gains.

The meta-policy combines gold-free retrieval-state features, five out-of-fold
MLP votes, and a transparent score-rule vote.  Hyperparameters and STOP
thresholds are selected only from train out-of-fold predictions.  Validation
labels are not inspected until the final model ensemble has been frozen.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from experiments.evaluate_stop_value_ensemble import (
    MODEL_NAMES,
    _load_answers,
    _load_prediction_sets,
    _prediction_paths,
    _rule_matrix,
    _vote_matrix,
)
from experiments.evaluate_stop_value_policy import _load_rows
from experiments.train_answer_policy import _bucket
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl
from experiments.train_stop_policy import ACTIONS, MAX_SEARCH_STEPS
from experiments.train_stop_value_policy import _state_examples


@dataclass(frozen=True)
class TreeConfig:
    min_samples_leaf: int
    max_features: float


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable ExtraTrees STOP gain ensemble")
    parser.add_argument(
        "--train-answer-root",
        default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_train_1000",
    )
    parser.add_argument(
        "--train-trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_train_1000",
    )
    parser.add_argument(
        "--validation-answer-root",
        default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_validation_2000",
    )
    parser.add_argument(
        "--validation-trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--max-train-rows-per-dataset", type=int, default=1000)
    parser.add_argument("--max-validation-rows-per-dataset", type=int, default=2000)
    parser.add_argument("--evidence-characters", type=int, default=1400)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--leaf-sizes", nargs="+", type=int, default=(5, 10, 20, 40))
    parser.add_argument("--max-features", nargs="+", type=float, default=(0.7, 1.0))
    parser.add_argument("--ensemble-seeds", nargs="+", type=int, default=(42, 43, 44, 45, 46))
    parser.add_argument("--positive-weight", type=float, default=6.0)
    parser.add_argument("--gain-weight", type=float, default=5.0)
    parser.add_argument("--search-cost", type=float, default=0.005)
    parser.add_argument("--f1-weight", type=float, default=0.5)
    parser.add_argument("--em-weight", type=float, default=0.5)
    parser.add_argument("--dataset-harm-penalty", type=float, default=2.0)
    parser.add_argument("--threshold-steps", type=int, default=61)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--train-rule-predictions",
        default="data/benchmarks/multihop/rule_policy/retrieval_score_saturation_train_1000_oof_predictions.jsonl",
    )
    parser.add_argument(
        "--validation-rule-predictions",
        default="data/benchmarks/multihop/rule_policy/retrieval_score_saturation_validation_2000_predictions.jsonl",
    )
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-out", required=True)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    total = args.f1_weight + args.em_weight
    if total <= 0 or args.search_cost < 0 or args.dataset_harm_penalty < 0:
        parser.error("utility 가중치와 비용을 확인하세요.")
    if args.folds < 3 or args.trees < 10 or args.threshold_steps < 3:
        parser.error("folds/trees/threshold steps가 너무 작습니다.")
    if any(value < 1 for value in args.leaf_sizes) or any(not 0 < value <= 1 for value in args.max_features):
        parser.error("leaf size와 max features 범위를 확인하세요.")
    args.f1_weight /= total
    args.em_weight /= total
    return args


def _state_inputs(args, split: str):
    proxy = argparse.Namespace(
        answer_root=args.train_answer_root if split == "train" else args.validation_answer_root,
        trajectory_root=args.train_trajectory_root if split == "train" else args.validation_trajectory_root,
        index_root=args.index_root,
        datasets=args.datasets,
        split=split,
        max_rows_per_dataset=(
            args.max_train_rows_per_dataset if split == "train" else args.max_validation_rows_per_dataset
        ),
    )
    rows, trajectories, contents, graph_indices = _load_rows(proxy)
    states = _state_examples(
        rows,
        trajectories,
        contents,
        graph_indices,
        args.evidence_characters,
        "bellman",
        args.f1_weight,
        args.em_weight,
        args.search_cost,
    )
    return rows, states


def _meta_features(
    rows: list[dict],
    states,
    votes: np.ndarray,
    rule: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    state_by_key = {(state.question_id, state.step): state for state in states}
    features, targets, steps, keys = [], [], [], []
    for row_index, row in enumerate(rows):
        row_id = str(row["id"])
        for step in range(MAX_SEARCH_STEPS):
            state = state_by_key[(row_id, step)]
            vote_fraction = float(votes[row_index, step]) / len(MODEL_NAMES)
            agreement = abs(vote_fraction - 0.5) * 2.0
            feature = (
                *state.numeric,
                vote_fraction,
                agreement,
                float(rule[row_index, step]),
            )
            features.append(feature)
            targets.append(float(state.advantage))
            steps.append(step)
            keys.append((row_id, step))
    return (
        np.asarray(features, dtype="float32"),
        np.asarray(targets, dtype="float64"),
        np.asarray(steps, dtype="int64"),
        keys,
    )


def _fit_predict_oof(
    rows: list[dict],
    features: np.ndarray,
    targets: np.ndarray,
    steps: np.ndarray,
    keys: list[tuple[str, int]],
    config: TreeConfig,
    args,
) -> np.ndarray:
    output = np.zeros(len(features), dtype="float64")
    row_fold = {str(row["id"]): _bucket(str(row["id"]), "stable-stop-tree", args.folds) for row in rows}
    for step in range(MAX_SEARCH_STEPS):
        step_indices = np.flatnonzero(steps == step)
        for fold in range(args.folds):
            test = np.asarray([index for index in step_indices if row_fold[keys[index][0]] == fold], dtype="int64")
            fit = np.asarray([index for index in step_indices if row_fold[keys[index][0]] != fold], dtype="int64")
            model = _fit_model(features[fit], targets[fit], config, args, seed=args.bootstrap_seed + step * 100 + fold)
            output[test] = model.predict(features[test])
    return output


def _fit_model(features, targets, config: TreeConfig, args, seed: int):
    weights = 1.0 + args.gain_weight * np.abs(targets) + args.positive_weight * (targets > 0.0)
    model = ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(features, targets, sample_weight=weights)
    return model


def _prediction_matrix(rows: list[dict], predictions: np.ndarray, keys: list[tuple[str, int]]) -> np.ndarray:
    lookup = {key: float(value) for key, value in zip(keys, predictions, strict=True)}
    return np.asarray(
        [[lookup[(str(row["id"]), step)] for step in range(MAX_SEARCH_STEPS)] for row in rows],
        dtype="float64",
    )


def _simulate(predictions: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    stop_steps = np.full(len(predictions), MAX_SEARCH_STEPS, dtype="int64")
    active = np.ones(len(predictions), dtype="bool")
    for step in range(MAX_SEARCH_STEPS):
        continue_search = predictions[:, step] > thresholds[step]
        stopping = active & ~continue_search
        stop_steps[stopping] = step
        active &= continue_search
    return stop_steps


def _selected(rows: list[dict], stop_steps: np.ndarray, metric: str) -> np.ndarray:
    return np.asarray(
        [float(row["actions"][ACTIONS[int(step)]][metric]) for row, step in zip(rows, stop_steps, strict=True)]
    )


def _objective(rows: list[dict], stop_steps: np.ndarray, args) -> tuple[float, float, float]:
    em, f1 = _selected(rows, stop_steps, "em"), _selected(rows, stop_steps, "f1")
    utility = args.em_weight * em + args.f1_weight * f1 - args.search_cost * stop_steps
    direct = np.asarray(
        [args.em_weight * float(row["direct_em"]) + args.f1_weight * float(row["direct_f1"]) for row in rows]
    )
    policy_values, harms = [], []
    for dataset in args.datasets:
        mask = np.asarray([str(row["dataset"]) == dataset for row in rows])
        policy, baseline = float(utility[mask].mean()), float(direct[mask].mean())
        policy_values.append(policy)
        harms.append(max(0.0, baseline - policy))
    protected = float(np.mean(policy_values) - args.dataset_harm_penalty * sum(harms))
    return protected, float(f1.mean()), float(stop_steps.mean())


def _threshold_grids(predictions: np.ndarray, steps: int) -> tuple[np.ndarray, ...]:
    grids = []
    for step in range(MAX_SEARCH_STEPS):
        values = predictions[:, step]
        quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, steps)))
        grids.append(
            np.unique(
                np.concatenate(
                    ([np.nextafter(quantiles[0], -np.inf)], quantiles, [np.nextafter(quantiles[-1], np.inf)])
                )
            )
        )
    return tuple(grids)


def _calibrate(rows: list[dict], predictions: np.ndarray, args) -> tuple[float, ...]:
    grids = _threshold_grids(predictions, args.threshold_steps)

    def score(candidate):
        protected, f1, average_steps = _objective(rows, _simulate(predictions, candidate), args)
        return protected, f1, -average_steps, tuple(float(value) for value in candidate)

    starts = (
        tuple(float(grid[0]) for grid in grids),
        tuple(float(grid[len(grid) // 2]) for grid in grids),
        tuple(float(grid[-1]) for grid in grids),
    )
    solutions = []
    for start in starts:
        current = start
        for _ in range(5):
            previous = current
            for step in range(MAX_SEARCH_STEPS):
                candidates = []
                for threshold in grids[step]:
                    candidate = list(current)
                    candidate[step] = float(threshold)
                    candidates.append(score(candidate))
                current = max(candidates)[-1]
            if current == previous:
                break
        solutions.append(score(current))
    return max(solutions)[-1]


def _summary(rows: list[dict], stop_steps: np.ndarray, datasets: tuple[str, ...]) -> dict:
    result = {}
    for dataset in datasets:
        indices = [index for index, row in enumerate(rows) if str(row["dataset"]) == dataset]
        selected_rows = [rows[index] for index in indices]
        selected_steps = stop_steps[indices]
        result[dataset] = {
            "n": len(indices),
            "em": float(_selected(selected_rows, selected_steps, "em").mean()),
            "f1": float(_selected(selected_rows, selected_steps, "f1").mean()),
            "average_extra_searches": float(selected_steps.mean()),
            "stop_steps": {
                str(step): int(np.sum(selected_steps == step)) for step in range(MAX_SEARCH_STEPS + 1)
            },
        }
    return result


def _records(rows, predictions, thresholds):
    stop_steps = _simulate(predictions, thresholds)
    records = []
    for row_index, (row, step) in enumerate(zip(rows, stop_steps, strict=True)):
        action = ACTIONS[int(step)]
        selected = row["actions"][action]
        records.append(
            {
                "id": str(row["id"]),
                "dataset": str(row["dataset"]),
                "selected_action": action,
                "stop_step": int(step),
                "predicted_gains": predictions[row_index].tolist(),
                "thresholds": list(thresholds),
                "direct_em": float(row["direct_em"]),
                "direct_f1": float(row["direct_f1"]),
                "policy_em": float(selected["em"]),
                "policy_f1": float(selected["f1"]),
                "source_recall": float(selected["source_recall"]),
                "all_support_recall": float(selected["all_support_recall"]),
                "oracle_f1": float(row["oracle_f1"]),
            }
        )
    return records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in sorted(rows, key=lambda item: (item["dataset"], item["id"])):
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    manifests, train_prediction_paths, validation_prediction_paths = _prediction_paths()
    train_rows, train_states = _state_inputs(args, "train")
    validation_rows, validation_states = _state_inputs(args, "validation")
    expected_train = {str(row["id"]) for row in train_rows}
    expected_validation = {str(row["id"]) for row in validation_rows}
    train_policy_predictions = _load_prediction_sets(train_prediction_paths, expected_train)
    validation_policy_predictions = _load_prediction_sets(validation_prediction_paths, expected_validation)
    final_thresholds = tuple(
        tuple(float(value) for value in json.loads(path.read_text(encoding="utf-8"))["threshold"])
        for path in manifests
    )
    train_votes = _vote_matrix(train_rows, train_policy_predictions)
    validation_votes = _vote_matrix(validation_rows, validation_policy_predictions, final_thresholds)
    train_rule = _rule_matrix(train_rows, Path(args.train_rule_predictions))
    validation_rule = _rule_matrix(validation_rows, Path(args.validation_rule_predictions))
    train_features, train_targets, train_steps, train_keys = _meta_features(
        train_rows, train_states, train_votes, train_rule
    )
    validation_features, _, validation_steps, validation_keys = _meta_features(
        validation_rows, validation_states, validation_votes, validation_rule
    )

    candidate_audit = []
    selected = None
    for leaf_size, max_features in itertools.product(args.leaf_sizes, args.max_features):
        config = TreeConfig(leaf_size, max_features)
        oof_flat = _fit_predict_oof(
            train_rows, train_features, train_targets, train_steps, train_keys, config, args
        )
        oof = _prediction_matrix(train_rows, oof_flat, train_keys)
        thresholds = _calibrate(train_rows, oof, args)
        stop_steps = _simulate(oof, thresholds)
        protected, f1, average_steps = _objective(train_rows, stop_steps, args)
        audit = {
            "config": asdict(config),
            "thresholds": list(thresholds),
            "protected_utility": protected,
            "f1": f1,
            "average_extra_searches": average_steps,
            "metrics": _summary(train_rows, stop_steps, args.datasets),
        }
        candidate_audit.append(audit)
        key = (protected, f1, -average_steps, -leaf_size, max_features)
        if selected is None or key > selected[0]:
            selected = (key, config, thresholds, audit)
        print(
            f"candidate leaf={leaf_size}, features={max_features:.2f} | "
            f"OOF F1={f1:.4f}, extra={average_steps:.3f}, thresholds={thresholds}",
            flush=True,
        )
    assert selected is not None
    _, config, thresholds, selected_audit = selected

    validation_predictions = np.zeros((len(validation_rows), MAX_SEARCH_STEPS), dtype="float64")
    final_models = []
    for seed in args.ensemble_seeds:
        seed_models = []
        flat = np.zeros(len(validation_features), dtype="float64")
        for step in range(MAX_SEARCH_STEPS):
            fit = np.flatnonzero(train_steps == step)
            test = np.flatnonzero(validation_steps == step)
            model = _fit_model(train_features[fit], train_targets[fit], config, args, seed + step * 100)
            flat[test] = model.predict(validation_features[test])
            seed_models.append(model)
        validation_predictions += _prediction_matrix(validation_rows, flat, validation_keys)
        final_models.append(seed_models)
    validation_predictions /= len(args.ensemble_seeds)
    records = _records(validation_rows, validation_predictions, thresholds)
    stop_steps = np.asarray([row["stop_step"] for row in records], dtype="int64")

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "format_version": 1,
            "policy": "stable_rare_gain_extra_trees_ensemble",
            "config": asdict(config),
            "thresholds": thresholds,
            "models": final_models,
            "feature_count": train_features.shape[1],
            "ensemble_seeds": tuple(args.ensemble_seeds),
        },
        model_path,
    )
    _write_jsonl(Path(args.predictions_out), records)
    result = {
        "format_version": 1,
        "policy": "stable_rare_gain_extra_trees_ensemble",
        "validation_labels_used_for_selection": False,
        "selected_config": asdict(config),
        "thresholds": list(thresholds),
        "ensemble_seeds": list(args.ensemble_seeds),
        "feature_count": int(train_features.shape[1]),
        "utility": {
            "f1_weight": args.f1_weight,
            "em_weight": args.em_weight,
            "search_cost": args.search_cost,
            "dataset_harm_penalty": args.dataset_harm_penalty,
        },
        "selected_train_oof": selected_audit,
        "train_oof_candidates": candidate_audit,
        "validation": _summary(validation_rows, stop_steps, args.datasets),
        "model": str(model_path),
        "predictions": args.predictions_out,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
