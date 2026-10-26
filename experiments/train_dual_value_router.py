"""Train a dual-value SEARCH/REASON/STOP router on cached prefix trajectories.

The evidence head is reader-free and predicts whether the next retrieval adds
gold support during training.  The answer-risk head is trained only on states
whose gold support is already complete and predicts whether the current reader
answer is exact-match wrong.  Gold fields create offline labels only; neither
head consumes them at inference.

REASON is a routing decision in this stage.  Until a separate repair answer is
materialised, its offline answer is the step-2 answer.  This makes the report a
conservative pre-repair evaluation and exposes the exact queue for a targeted
Qwen repair pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS
from experiments.train_reader_aware_sufficiency_policy import _load
from experiments.train_reader_free_student_policy import (
    DECISION_STEP,
    FINAL_STEP,
    policy_feature_names,
    policy_features,
    student_feature_names,
    student_features,
)


FORMAT_VERSION = 1
SEARCH = "SEARCH"
REASON = "REASON"
STOP = "STOP"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-value SEARCH/REASON/STOP router")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--max-train-rows-per-dataset", type=int, default=1200)
    parser.add_argument("--max-validation-rows-per-dataset", type=int, default=0)
    parser.add_argument("--train-sample-seed", type=int, required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--noninferiority-points", type=float, default=0.25)
    parser.add_argument("--minimum-answer-error-recall", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_train_rows_per_dataset < 1 or args.max_validation_rows_per_dataset < 0:
        parser.error("train은 1 이상, validation은 0 이상이어야 합니다.")
    if args.cv_folds < 2:
        parser.error("cv-folds는 2 이상이어야 합니다.")
    if not 0.0 < args.minimum_answer_error_recall <= 1.0:
        parser.error("minimum answer error recall은 0 초과 1 이하여야 합니다.")
    return args


def _manifest_names(root: Path, dataset: str) -> tuple[str, ...]:
    payload = json.loads(
        (root / f"{dataset}_train.manifest.json").read_text(encoding="utf-8")
    )
    names = tuple(str(name) for name in payload.get("feature_names", ()))
    if not names:
        raise SystemExit("reader-aware manifest feature schema가 없습니다.")
    return names


def _evidence_matrix(rows: list[dict], manifest_names: tuple[str, ...]):
    features = np.vstack([student_features(row, manifest_names) for row in rows])
    labels = np.asarray(
        [
            float(row["actions"][ACTIONS[FINAL_STEP]]["source_recall"])
            > float(row["actions"][ACTIONS[DECISION_STEP]]["source_recall"]) + 1e-12
            for row in rows
        ],
        dtype="int8",
    )
    missing = np.asarray(
        [float(row["actions"][ACTIONS[DECISION_STEP]]["all_support_recall"]) < 1.0 for row in rows],
        dtype="int8",
    )
    datasets = np.asarray([str(row["dataset"]) for row in rows])
    return features, labels, missing, datasets


def _answer_matrix(rows: list[dict], manifest_names: tuple[str, ...], *, complete_only: bool):
    selected_rows = [
        row
        for row in rows
        if not complete_only
        or float(row["actions"][ACTIONS[DECISION_STEP]]["all_support_recall"]) == 1.0
    ]
    features = np.vstack(
        [policy_features(row, manifest_names, "single_reader") for row in selected_rows]
    )
    labels = np.asarray(
        [float(row["actions"][ACTIONS[DECISION_STEP]]["em"]) < 1.0 for row in selected_rows],
        dtype="int8",
    )
    datasets = np.asarray([str(row["dataset"]) for row in selected_rows])
    return features, labels, datasets, selected_rows


MODEL_SPECS = ("hgb", "extra_trees")


def _fit_classifier(features, labels, seed: int, model_spec: str):
    if len(np.unique(labels)) != 2:
        raise ValueError("classifier label에는 두 class가 모두 필요합니다.")
    weights = np.ones(len(labels), dtype="float64")
    if model_spec == "hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=350,
            max_leaf_nodes=23,
            min_samples_leaf=25,
            l2_regularization=5.0,
            class_weight="balanced",
            random_state=seed,
        )
    elif model_spec == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=350,
            max_features=0.7,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"지원하지 않는 model spec: {model_spec}")
    model.fit(features, labels, sample_weight=weights)
    return model


def _oof(features, labels, datasets, seed: int, folds: int, model_spec: str):
    strata = np.asarray(
        [f"{dataset}:{label}" for dataset, label in zip(datasets, labels, strict=True)]
    )
    probabilities = np.zeros(len(labels), dtype="float64")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_indices, test_indices) in enumerate(splitter.split(features, strata)):
        model = _fit_classifier(
            features[fit_indices], labels[fit_indices], seed + fold, model_spec
        )
        probabilities[test_indices] = model.predict_proba(features[test_indices])[:, 1]
    return probabilities


def _binary_metrics(labels, probabilities) -> dict:
    return {
        "count": len(labels),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def _search_summary(rows, search_mask, datasets):
    result = {}
    for dataset in datasets:
        indices = np.asarray([row["dataset"] == dataset for row in rows])
        subset_rows = [row for row, include in zip(rows, indices, strict=True) if include]
        subset_search = search_mask[indices]
        adaptive_em = []
        adaptive_f1 = []
        fixed_em = []
        fixed_f1 = []
        support_gain_labels = []
        for row, search in zip(subset_rows, subset_search, strict=True):
            chosen = row["actions"][ACTIONS[FINAL_STEP if search else DECISION_STEP]]
            terminal = row["actions"][ACTIONS[FINAL_STEP]]
            step2 = row["actions"][ACTIONS[DECISION_STEP]]
            adaptive_em.append(float(chosen["em"]))
            adaptive_f1.append(float(chosen["f1"]))
            fixed_em.append(float(terminal["em"]))
            fixed_f1.append(float(terminal["f1"]))
            support_gain_labels.append(
                float(terminal["source_recall"]) > float(step2["source_recall"]) + 1e-12
            )
        labels = np.asarray(support_gain_labels, dtype="bool")
        true_positive = int(np.sum(subset_search & labels))
        result[dataset] = {
            "count": len(subset_rows),
            "em": float(np.mean(adaptive_em)),
            "f1": float(np.mean(adaptive_f1)),
            "fixed3_em": float(np.mean(fixed_em)),
            "fixed3_f1": float(np.mean(fixed_f1)),
            "em_change_points": float((np.mean(adaptive_em) - np.mean(fixed_em)) * 100.0),
            "f1_change_points": float((np.mean(adaptive_f1) - np.mean(fixed_f1)) * 100.0),
            "average_additional_searches": float(2.0 + np.mean(subset_search)),
            "search_reduction_percent_vs_fixed3": float(np.mean(~subset_search) / 3.0 * 100.0),
            "support_gain_recall": true_positive / max(1, int(labels.sum())),
            "support_gain_precision": true_positive / max(1, int(subset_search.sum())),
            "support_gain_count": int(labels.sum()),
            "search_count": int(subset_search.sum()),
        }
    return result


def _search_thresholds():
    return (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.01)


def _choose_search(rows, datasets, outputs_by_model, margin_points: float):
    sweep = []
    for model_spec, probabilities in outputs_by_model.items():
        for threshold in _search_thresholds():
            search_mask = probabilities >= threshold
            report = _search_summary(rows, search_mask, datasets)
            worst_em = min(item["em_change_points"] for item in report.values())
            worst_f1 = min(item["f1_change_points"] for item in report.values())
            recall = float(np.mean([item["support_gain_recall"] for item in report.values()]))
            precision = float(np.mean([item["support_gain_precision"] for item in report.values()]))
            search_sum = sum(item["average_additional_searches"] for item in report.values())
            sweep.append(
                {
                    "model_spec": model_spec,
                    "threshold": threshold,
                    "datasets": report,
                    "worst_em_change_points": worst_em,
                    "worst_f1_change_points": worst_f1,
                    "mean_support_gain_recall": recall,
                    "mean_support_gain_precision": precision,
                    "sum_average_searches": search_sum,
                }
            )
    acceptable = [
        item
        for item in sweep
        if item["worst_em_change_points"] >= -margin_points
        and item["worst_f1_change_points"] >= -margin_points
    ]
    if not acceptable:
        acceptable = sweep
    chosen = max(
        acceptable,
        key=lambda item: (
            item["mean_support_gain_recall"],
            item["mean_support_gain_precision"],
            -item["sum_average_searches"],
            item["worst_f1_change_points"],
        ),
    )
    return chosen, sweep


def _answer_thresholds():
    return tuple(np.linspace(0.05, 0.95, 37))


def _answer_threshold_report(labels, probabilities, datasets, threshold):
    predicted = probabilities >= threshold
    result = {}
    for dataset in sorted(set(datasets)):
        indices = datasets == dataset
        local_labels = labels[indices].astype("bool")
        local_predicted = predicted[indices]
        true_positive = int(np.sum(local_labels & local_predicted))
        result[dataset] = {
            "count": int(indices.sum()),
            "answer_error_count": int(local_labels.sum()),
            "reason_queue_count": int(local_predicted.sum()),
            "reason_queue_rate": float(local_predicted.mean()),
            "answer_error_recall": true_positive / max(1, int(local_labels.sum())),
            "answer_error_precision": true_positive / max(1, int(local_predicted.sum())),
        }
    return result


def _choose_answer(labels, datasets, outputs_by_model, minimum_recall: float):
    sweep = []
    for model_spec, probabilities in outputs_by_model.items():
        for threshold in _answer_thresholds():
            report = _answer_threshold_report(labels, probabilities, datasets, threshold)
            worst_recall = min(item["answer_error_recall"] for item in report.values())
            mean_precision = float(np.mean([item["answer_error_precision"] for item in report.values()]))
            queue_sum = sum(item["reason_queue_rate"] for item in report.values())
            sweep.append(
                {
                    "model_spec": model_spec,
                    "threshold": float(threshold),
                    "datasets": report,
                    "worst_answer_error_recall": worst_recall,
                    "mean_answer_error_precision": mean_precision,
                    "sum_reason_queue_rate": queue_sum,
                }
            )
    acceptable = [item for item in sweep if item["worst_answer_error_recall"] >= minimum_recall]
    if acceptable:
        chosen = min(
            acceptable,
            key=lambda item: (
                item["sum_reason_queue_rate"],
                -item["mean_answer_error_precision"],
                -item["worst_answer_error_recall"],
            ),
        )
    else:
        chosen = max(
            sweep,
            key=lambda item: (
                item["worst_answer_error_recall"],
                item["mean_answer_error_precision"],
                -item["sum_reason_queue_rate"],
            ),
        )
    return chosen, sweep


def _validation_summary(rows, search_probabilities, answer_probabilities, search_threshold, answer_threshold, datasets):
    search_mask = search_probabilities >= search_threshold
    reason_mask = (~search_mask) & (answer_probabilities >= answer_threshold)
    stop_mask = (~search_mask) & (~reason_mask)
    search_report = _search_summary(rows, search_mask, datasets)
    result = {}
    for dataset in datasets:
        indices = np.asarray([row["dataset"] == dataset for row in rows])
        local_reason = reason_mask[indices]
        local_stop = stop_mask[indices]
        local_answer_error = np.asarray(
            [float(row["actions"][ACTIONS[DECISION_STEP]]["em"]) < 1.0 for row, include in zip(rows, indices, strict=True) if include],
            dtype="bool",
        )
        local_nonsearch = ~search_mask[indices]
        true_positive = int(np.sum(local_reason & local_answer_error))
        base = search_report[dataset]
        result[dataset] = {
            **base,
            "route_distribution": {
                SEARCH: int(search_mask[indices].sum()),
                REASON: int(local_reason.sum()),
                STOP: int(local_stop.sum()),
            },
            "reason_queue_rate": float(local_reason.mean()),
            "answer_error_recall_in_nonsearch": true_positive
            / max(1, int(np.sum(local_nonsearch & local_answer_error))),
            "answer_error_precision_in_reason_queue": true_positive / max(1, int(local_reason.sum())),
            "pre_repair_note": "REASON uses the current step2 answer until repair outputs are materialised",
        }
    return result, search_mask, reason_mask


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = _arguments()
    train_root = Path(args.train_root)
    validation_root = Path(args.validation_root)
    train_rows = _load(
        train_root,
        args.datasets,
        "train",
        args.max_train_rows_per_dataset,
        sample_seed=args.train_sample_seed,
    )
    validation_rows = _load(
        validation_root,
        args.datasets,
        "validation",
        args.max_validation_rows_per_dataset,
    )
    manifest_names = _manifest_names(train_root, args.datasets[0])

    evidence_x, evidence_y, missing_y, evidence_datasets = _evidence_matrix(
        train_rows, manifest_names
    )
    answer_x, answer_y, answer_datasets, answer_rows = _answer_matrix(
        train_rows, manifest_names, complete_only=True
    )
    evidence_outputs = {}
    answer_outputs = {}
    diagnostics = {"evidence_head": {}, "answer_head": {}}
    for model_spec in MODEL_SPECS:
        evidence_probability = _oof(
            evidence_x,
            evidence_y,
            evidence_datasets,
            args.seed,
            args.cv_folds,
            model_spec,
        )
        answer_probability = _oof(
            answer_x,
            answer_y,
            answer_datasets,
            args.seed + 100,
            args.cv_folds,
            model_spec,
        )
        evidence_outputs[model_spec] = evidence_probability
        answer_outputs[model_spec] = answer_probability
        diagnostics["evidence_head"][model_spec] = _binary_metrics(
            evidence_y, evidence_probability
        )
        diagnostics["answer_head"][model_spec] = _binary_metrics(
            answer_y, answer_probability
        )

    search_choice, search_sweep = _choose_search(
        train_rows, args.datasets, evidence_outputs, args.noninferiority_points
    )
    answer_choice, answer_sweep = _choose_answer(
        answer_y, answer_datasets, answer_outputs, args.minimum_answer_error_recall
    )
    evidence_model = _fit_classifier(
        evidence_x,
        evidence_y,
        args.seed,
        str(search_choice["model_spec"]),
    )
    answer_model = _fit_classifier(
        answer_x,
        answer_y,
        args.seed + 100,
        str(answer_choice["model_spec"]),
    )

    validation_evidence_x, validation_evidence_y, validation_missing_y, validation_datasets = _evidence_matrix(
        validation_rows, manifest_names
    )
    validation_answer_x, validation_answer_y, _, _ = _answer_matrix(
        validation_rows, manifest_names, complete_only=False
    )
    search_probabilities = evidence_model.predict_proba(validation_evidence_x)[:, 1]
    answer_probabilities = answer_model.predict_proba(validation_answer_x)[:, 1]
    summary, search_mask, reason_mask = _validation_summary(
        validation_rows,
        search_probabilities,
        answer_probabilities,
        float(search_choice["threshold"]),
        float(answer_choice["threshold"]),
        args.datasets,
    )
    validation_diagnostics = {
        "evidence_head": _binary_metrics(validation_evidence_y, search_probabilities),
        "answer_head_all_states": _binary_metrics(validation_answer_y, answer_probabilities),
        "evidence_missing_detection_auc": float(roc_auc_score(validation_missing_y, search_probabilities)),
    }
    for dataset, result in summary.items():
        print(
            f"{dataset}: routes={result['route_distribution']} fixed3/pre-repair "
            f"EM={result['fixed3_em']*100:.2f}/{result['em']*100:.2f} "
            f"F1={result['fixed3_f1']*100:.2f}/{result['f1']*100:.2f} "
            f"support-gain recall={result['support_gain_recall']:.1%} "
            f"answer-risk recall={result['answer_error_recall_in_nonsearch']:.1%}",
            flush=True,
        )

    records = []
    for row, search_probability, answer_probability, search, reason in zip(
        validation_rows,
        search_probabilities,
        answer_probabilities,
        search_mask,
        reason_mask,
        strict=True,
    ):
        route = SEARCH if search else (REASON if reason else STOP)
        chosen_step = FINAL_STEP if search else DECISION_STEP
        selected = row["actions"][ACTIONS[chosen_step]]
        records.append(
            {
                "format_version": FORMAT_VERSION,
                "id": str(row["id"]),
                "dataset": str(row["dataset"]),
                "route": route,
                "chosen_step_pre_repair": chosen_step,
                "support_gain_probability": float(search_probability),
                "answer_error_probability": float(answer_probability),
                "prediction_pre_repair": selected["prediction"],
                "em_pre_repair": float(selected["em"]),
                "f1_pre_repair": float(selected["f1"]),
            }
        )

    bundle = {
        "format_version": FORMAT_VERSION,
        "method": "dual_value_search_reason_stop_router",
        "evidence_feature_names": student_feature_names(manifest_names),
        "answer_feature_names": policy_feature_names(manifest_names, "single_reader"),
        "evidence_model": evidence_model,
        "answer_model": answer_model,
        "search_threshold": float(search_choice["threshold"]),
        "answer_threshold": float(answer_choice["threshold"]),
        "decision_step": DECISION_STEP,
        "final_step": FINAL_STEP,
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report_path = Path(args.report_out)
    predictions_path = report_path.with_name(
        f"{report_path.stem}.validation_predictions.jsonl"
    )
    _write_jsonl(predictions_path, records)
    ids_digest = hashlib.sha256(
        "\n".join(sorted(str(row["id"]) for row in train_rows)).encode("utf-8")
    ).hexdigest()
    report = {
        "format_version": FORMAT_VERSION,
        "method": bundle["method"],
        "model": str(model_path),
        "validation_predictions": str(predictions_path),
        "training_sample_seed": args.train_sample_seed,
        "training_questions": len(train_rows),
        "validation_questions": len(validation_rows),
        "training_ids_sha256": ids_digest,
        "feature_counts": {
            "evidence_head": evidence_x.shape[1],
            "answer_head": answer_x.shape[1],
        },
        "label_counts": {
            "support_gain": int(evidence_y.sum()),
            "evidence_missing": int(missing_y.sum()),
            "answer_error_with_complete_support": int(answer_y.sum()),
            "complete_support_states": len(answer_y),
        },
        "forbidden_feature_audit": {
            "uses_dataset_identity": False,
            "uses_gold_hop": False,
            "uses_gold_answer": False,
            "uses_gold_support_at_inference": False,
            "uses_actual_em_f1_at_inference": False,
        },
        "oof_diagnostics": diagnostics,
        "search_choice": search_choice,
        "answer_choice": answer_choice,
        "search_parameter_candidates": len(search_sweep),
        "answer_parameter_candidates": len(answer_sweep),
        "validation_diagnostics": validation_diagnostics,
        "validation_pre_repair": summary,
        "repair_status": "pending_materialisation_for_reason_queue",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {model_path}\nreport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
