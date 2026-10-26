"""Learn dataset-blind proof width from question and evidence-score structure."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.train_counterfactual_proof_policy import (
    _candidates,
    _group_indices,
    _load_split,
    _support_metrics,
)
from experiments.train_multihop_policy import SUPPORTED_DATASETS
from src.causal_evidence import WIDTH_FEATURE_NAMES, evidence_width_features, proof_policy_features


FORMAT_VERSION = 1


def _integer_counts(values) -> dict[int, int]:
    return dict(sorted((int(key), int(value)) for key, value in Counter(values).items()))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question-specific evidence width policy")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument(
        "--train-trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_train_1000",
    )
    parser.add_argument(
        "--validation-trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument("--support-model", default="data/models/counterfactual_proof_completion_policy.joblib")
    parser.add_argument("--model-out", default="data/models/evidence_width_policy.joblib")
    parser.add_argument("--report-out", default="data/models/evidence_width_policy.report.json")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-train-rows-per-dataset", type=int, default=1000)
    parser.add_argument("--max-validation-rows-per-dataset", type=int, default=2000)
    parser.add_argument("--minimum-width", type=int, default=2)
    parser.add_argument("--maximum-width", type=int, default=8)
    parser.add_argument("--quantile", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if not 0.5 <= args.quantile < 1.0 or not 1 <= args.minimum_width <= args.maximum_width <= 8:
        parser.error("quantile 또는 width 범위를 확인하세요.")
    return args


def _target_width(candidates, groups, probabilities, minimum: int, maximum: int) -> tuple[int, bool]:
    ranking = np.argsort(-probabilities)
    rank_by_index = {int(index): rank + 1 for rank, index in enumerate(ranking)}
    required_ranks = []
    external = False
    for group in groups:
        if not group:
            external = True
            continue
        required_ranks.append(min(rank_by_index[index] for index in group))
    target = maximum if external or not required_ranks else max(minimum, max(required_ranks))
    return min(maximum, int(target)), external


def _feature_rows(args, split: str, trajectory_root: Path, maximum: int, support_model):
    rows = []
    for dataset in args.datasets:
        joined, documents = _load_split(args, dataset, split, trajectory_root, maximum)
        for trajectory, example in joined:
            candidates = _candidates(trajectory)
            action_matrix = proof_policy_features(example.question, candidates, documents, ())
            probabilities = support_model.predict_proba(action_matrix)[:, 1][:-1]
            groups = _group_indices(example, candidates)
            target, external = _target_width(
                candidates,
                groups,
                probabilities,
                args.minimum_width,
                args.maximum_width,
            )
            rows.append(
                {
                    "id": example.policy_id,
                    "dataset": dataset,
                    "question": example.question,
                    "reasoning_type": example.reasoning_type,
                    "candidates": candidates,
                    "groups": groups,
                    "probabilities": probabilities,
                    "features": evidence_width_features(
                        support_model, example.question, candidates, documents
                    ),
                    "target_width": target,
                    "requires_external_search": external,
                }
            )
    return rows


def _evaluate(rows, model, minimum: int, maximum: int):
    result = {}
    predictions = []
    for dataset in SUPPORTED_DATASETS:
        selected_rows = [row for row in rows if row["dataset"] == dataset]
        if not selected_rows:
            continue
        for row in selected_rows:
            raw_width = float(model.predict(row["features"].reshape(1, -1))[0])
            width = int(np.clip(np.ceil(raw_width), minimum, maximum))
            ranking = np.argsort(-row["probabilities"])
            selected = set(int(index) for index in ranking[:width])
            metrics = _support_metrics(selected, row["groups"])
            predictions.append(
                {
                    "id": row["id"],
                    "dataset": dataset,
                    "predicted_width": width,
                    "target_width": row["target_width"],
                    "requires_external_search": row["requires_external_search"],
                    "source_recall": metrics[0],
                    "all_support_recall": metrics[1],
                    "support_precision": metrics[2],
                    "selected_indices": [int(index) for index in ranking[:width]],
                }
            )
        current = [row for row in predictions if row["dataset"] == dataset]
        result[dataset] = {
            "n": len(current),
            "predicted_width": float(np.mean([row["predicted_width"] for row in current])),
            "target_width": float(np.mean([row["target_width"] for row in current])),
            "exact_width": float(np.mean([row["predicted_width"] == row["target_width"] for row in current])),
            "external_search_rate": float(np.mean([row["requires_external_search"] for row in current])),
            "source_recall": float(np.mean([row["source_recall"] for row in current])),
            "all_support_recall": float(np.mean([row["all_support_recall"] for row in current])),
            "support_precision": float(np.mean([row["support_precision"] for row in current])),
            "width_distribution": dict(sorted(Counter(row["predicted_width"] for row in current).items())),
        }
        print(json.dumps({dataset: result[dataset]}, ensure_ascii=False, indent=2), flush=True)
    return result, predictions


def main() -> None:
    args = _arguments()
    support_bundle = joblib.load(args.support_model)
    support_model = support_bundle["model"]
    train_rows = _feature_rows(
        args,
        "train",
        Path(args.train_trajectory_root),
        args.max_train_rows_per_dataset,
        support_model,
    )
    features = np.vstack([row["features"] for row in train_rows])
    targets = np.asarray([row["target_width"] for row in train_rows], dtype="float64")
    dataset_counts = Counter(row["dataset"] for row in train_rows)
    weights = np.asarray([len(train_rows) / (len(dataset_counts) * dataset_counts[row["dataset"]]) for row in train_rows])
    print(
        f"train={len(train_rows):,}, target={_integer_counts(targets.astype(int))}, "
        f"external={np.mean([row['requires_external_search'] for row in train_rows])*100:.2f}%",
        flush=True,
    )
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=args.quantile,
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=args.seed,
    )
    model.fit(features, targets, sample_weight=weights)
    validation_rows = _feature_rows(
        args,
        "validation",
        Path(args.validation_trajectory_root),
        args.max_validation_rows_per_dataset,
        support_model,
    )
    validation, predictions = _evaluate(
        validation_rows, model, args.minimum_width, args.maximum_width
    )
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "format_version": FORMAT_VERSION,
            "method": "question_specific_evidence_width",
            "feature_names": WIDTH_FEATURE_NAMES,
            "minimum_width": args.minimum_width,
            "maximum_width": args.maximum_width,
            "quantile": args.quantile,
            "support_model": str(Path(args.support_model)),
            "model": model,
        },
        model_path,
    )
    report_path = Path(args.report_out)
    prediction_path = report_path.with_name(f"{report_path.stem}.predictions.jsonl")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as destination:
        for row in predictions:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report_path.write_text(
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "method": "question_specific_evidence_width",
                "training_target_distribution": _integer_counts(targets.astype(int)),
                "training_external_search_rate": float(np.mean([row["requires_external_search"] for row in train_rows])),
                "validation": validation,
                "model": str(model_path),
                "predictions": str(prediction_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {model_path}\nreport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
