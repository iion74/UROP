"""Train separate WHERE-to-repair and WHETHER-to-repair policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments.train_graph_repair_policy import (
    RepairState,
    _arguments as _legacy_arguments,
    _build_states,
    _state_metrics,
)
from src.proof_repair import REPAIR_FEATURE_NAMES, SYNDROME_FEATURE_NAMES, repair_syndrome_features


FORMAT_VERSION = 1


def _arguments() -> argparse.Namespace:
    args = _legacy_arguments()
    # Keep the legacy parser surface but write a separate factorized artifact.
    if args.model_out == "data/models/local_graph_proof_repair_policy.joblib":
        args.model_out = "data/models/factorized_graph_proof_repair_policy.joblib"
    if args.report_out == "data/models/local_graph_proof_repair_policy.report.json":
        args.report_out = "data/models/factorized_graph_proof_repair_policy.report.json"
    return args


def _fit_ranker(states: list[RepairState], seed: int):
    features = []
    labels = []
    weights = []
    used_states = 0
    for state in states:
        if state.tune or not state.labels.any():
            continue
        positive_count = int(state.labels.sum())
        negative_count = len(state.labels) - positive_count
        if not positive_count or not negative_count:
            continue
        used_states += 1
        features.append(state.features)
        labels.append(state.labels)
        weights.append(
            np.where(
                state.labels > 0,
                0.5 / positive_count,
                0.5 / negative_count,
            )
        )
    matrix = np.vstack(features)
    targets = np.concatenate(labels)
    sample_weights = np.concatenate(weights)
    ranker = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )
    ranker.fit(matrix, targets, sample_weight=sample_weights)
    return ranker, {
        "reachable_fit_states": used_states,
        "proposal_rows": len(targets),
        "positive_proposals": int(targets.sum()),
    }


def _rank_scores(ranker, state: RepairState) -> np.ndarray:
    return (
        ranker.predict_proba(state.features)[:, 1]
        if len(state.features)
        else np.empty(0, dtype="float64")
    )


def _syndrome_matrix(states: list[RepairState], ranker):
    return np.vstack(
        [
            repair_syndrome_features(
                state.features,
                _rank_scores(ranker, state),
                len(state.selected_ids),
            )
            for state in states
        ]
    )


def _fit_syndrome(states: list[RepairState], ranker, seed: int):
    fit_states = [state for state in states if not state.tune]
    features = _syndrome_matrix(fit_states, ranker)
    labels = np.asarray([float(state.labels.any()) for state in fit_states], dtype="int8")
    syndrome = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=2.0,
        class_weight="balanced",
        random_state=seed + 1,
    )
    syndrome.fit(features, labels)
    return syndrome, {
        "fit_states": len(labels),
        "positive_states": int(labels.sum()),
        "positive_rate": float(labels.mean()),
    }


def _evaluate(states, ranker, syndrome, threshold):
    state_features = _syndrome_matrix(states, ranker)
    syndrome_probabilities = syndrome.predict_proba(state_features)[:, 1]
    metrics = []
    base_metrics = []
    chosen_labels = []
    additions = []
    true_states = np.asarray([state.labels.any() for state in states], dtype="int8")
    gates = syndrome_probabilities >= threshold
    top1_correct = []
    for state, gate in zip(states, gates, strict=True):
        base_metrics.append(_state_metrics(state, ()))
        scores = _rank_scores(ranker, state)
        chosen = (int(np.argmax(scores)),) if gate and len(scores) else ()
        metrics.append(_state_metrics(state, chosen))
        additions.append(len(chosen))
        if chosen:
            chosen_labels.append(int(state.labels[chosen[0]]))
        if state.labels.any() and len(scores):
            top1_correct.append(int(state.labels[int(np.argmax(scores))]))
    base = np.asarray(base_metrics, dtype="float64")
    repaired = np.asarray(metrics, dtype="float64")
    true_positive = int(np.sum(gates & (true_states == 1)))
    predicted_positive = int(np.sum(gates))
    actual_positive = int(true_states.sum())
    return {
        "threshold": float(threshold),
        "baseline_source_recall": float(base[:, 0].mean()),
        "source_recall": float(repaired[:, 0].mean()),
        "baseline_all_support_recall": float(base[:, 1].mean()),
        "all_support_recall": float(repaired[:, 1].mean()),
        "baseline_support_precision": float(base[:, 2].mean()),
        "support_precision": float(repaired[:, 2].mean()),
        "support_f1": float(repaired[:, 3].mean()),
        "average_additions": float(np.mean(additions)),
        "repair_precision": float(np.mean(chosen_labels)) if chosen_labels else 1.0,
        "ranker_top1_recall_on_reachable_states": float(np.mean(top1_correct)) if top1_correct else 0.0,
        "syndrome_precision": true_positive / max(1, predicted_positive),
        "syndrome_recall": true_positive / max(1, actual_positive),
        "syndrome_gate_rate": predicted_positive / max(1, len(states)),
    }


def _choose_threshold(states, ranker, syndrome):
    tuning = [state for state in states if state.tune]
    candidates = np.unique(
        np.concatenate((np.linspace(0.1, 0.9, 33), np.asarray([0.925, 0.95, 0.975, 0.99])))
    )
    sweep = [_evaluate(tuning, ranker, syndrome, threshold) for threshold in candidates]
    for result in sweep:
        result["utility"] = (
            result["all_support_recall"]
            + 0.50 * result["source_recall"]
            + 0.25 * result["support_precision"]
            - 0.03 * result["average_additions"]
        )
    chosen = max(
        sweep,
        key=lambda result: (
            result["utility"],
            result["repair_precision"],
            result["threshold"],
        ),
    )
    return chosen, sweep


def _validation_report(states, ranker, syndrome, threshold):
    report = {}
    for dataset in dict.fromkeys(state.dataset for state in states):
        subset = [state for state in states if state.dataset == dataset]
        result = _evaluate(subset, ranker, syndrome, threshold)
        report[dataset] = result
        print(
            f"{dataset}: all-support={result['baseline_all_support_recall']*100:.2f}->"
            f"{result['all_support_recall']*100:.2f}, source={result['baseline_source_recall']*100:.2f}->"
            f"{result['source_recall']*100:.2f}, add={result['average_additions']:.2f}, "
            f"repair precision={result['repair_precision']*100:.2f}, "
            f"syndrome P/R={result['syndrome_precision']*100:.2f}/{result['syndrome_recall']*100:.2f}, "
            f"rank top1={result['ranker_top1_recall_on_reachable_states']*100:.2f}",
            flush=True,
        )
    return report


def main() -> None:
    args = _arguments()
    support_model = joblib.load(args.support_model)["model"]
    width_bundle = joblib.load(args.width_model)
    train_states, training_audit = _build_states(
        args,
        "train",
        Path(args.train_trajectory_root),
        args.max_train_rows_per_dataset,
        support_model,
        width_bundle,
    )
    ranker, ranker_fit = _fit_ranker(train_states, args.seed)
    syndrome, syndrome_fit = _fit_syndrome(train_states, ranker, args.seed)
    chosen, sweep = _choose_threshold(train_states, ranker, syndrome)
    print(
        f"chosen syndrome threshold={chosen['threshold']:.3f}, all-support="
        f"{chosen['baseline_all_support_recall']*100:.2f}->{chosen['all_support_recall']*100:.2f}, "
        f"add={chosen['average_additions']:.2f}, repair precision={chosen['repair_precision']*100:.2f}",
        flush=True,
    )
    validation_states, validation_audit = _build_states(
        args,
        "validation",
        Path(args.validation_trajectory_root),
        args.max_validation_rows_per_dataset,
        support_model,
        width_bundle,
    )
    validation = _validation_report(
        validation_states, ranker, syndrome, chosen["threshold"]
    )
    bundle = {
        "format_version": FORMAT_VERSION,
        "method": "factorized_local_graph_proof_repair",
        "ranker_feature_names": REPAIR_FEATURE_NAMES,
        "syndrome_feature_names": SYNDROME_FEATURE_NAMES,
        "ranker": ranker,
        "syndrome": syndrome,
        "syndrome_threshold": chosen["threshold"],
        "maximum_additions": 1,
        "neighbors_per_seed": args.neighbors_per_seed,
        "maximum_proposals": args.maximum_proposals,
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report = {
        "format_version": FORMAT_VERSION,
        "method": bundle["method"],
        "model": str(model_path),
        "training_audit": training_audit,
        "ranker_fit": ranker_fit,
        "syndrome_fit": syndrome_fit,
        "chosen_threshold": chosen,
        "threshold_sweep": sweep,
        "validation_audit": validation_audit,
        "validation": validation,
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {model_path}\nreport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
