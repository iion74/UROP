"""Cross-split check: train REASON benefit on all-reason holdout, test full validation.

This complements evaluate_reason_gain_gate.py but is exploratory: both cached
splits were inspected during earlier method development.  No Qwen calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from experiments.evaluate_answer_gain_search_policy import _read_dataset
from experiments.evaluate_reason_gain_gate import (
    DATASETS,
    HOLDOUT_ANALYSIS,
    HOLDOUT_ROOT,
    ROOT,
    TRAIN_ROOT,
    _judge_rows,
    _model,
    _oof,
    _predictions,
    _selected_features,
    _summary,
)
from experiments.train_reader_free_student_policy import policy_feature_names


OUTPUT = ROOT / "analysis" / "reason_gain_gate_full_validation_reverse.json"
MODEL_OUT = Path("data/models/reason_gain_gate_reverse.joblib")


def main() -> None:
    manifest = json.loads((HOLDOUT_ROOT / "hotpotqa_train.manifest.json").read_text())
    names = tuple(manifest["feature_names"])
    train_parts = []
    for dataset in DATASETS:
        data = _read_dataset(HOLDOUT_ROOT, dataset, "train", names, "single_reader")
        judged = _judge_rows([HOLDOUT_ANALYSIS / "all_reason_judge"], dataset, "train")
        if set(data["ids"]) != set(judged):
            raise ValueError(f"All-reason holdout coverage incomplete: {dataset}")
        gain = np.asarray([
            float(judged[identifier]["judged_f1"])
            - float(judged[identifier]["baseline_f1"])
            for identifier in data["ids"]
        ])
        train_parts.append({"ids": data["ids"], "x": data["x"], "gain": gain})
    x = np.vstack([part["x"] for part in train_parts])
    y = np.concatenate([part["gain"] for part in train_parts])
    strata = np.concatenate([
        np.asarray([f"{dataset}:{'positive' if value > 0 else 'negative' if value < 0 else 'zero'}"
                    for value in part["gain"]])
        for dataset, part in zip(DATASETS, train_parts, strict=True)
    ])
    oof_results = {}
    for kind in ("hgb", "extra_trees"):
        print(f"Cross-fitting {kind} on {len(y):,} all-REASON holdout examples", flush=True)
        prediction = _oof(x, y, strata, kind)
        rank = np.argsort(prediction, kind="stable")
        keep = np.zeros(len(y), dtype=bool)
        keep[rank[-round(len(y) * 0.5):]] = True
        oof_results[kind] = {
            "mse": float(np.mean((prediction - y) ** 2)),
            "rank50_advantage_vs_random_points": float(
                (np.mean(keep * y) - 0.5 * np.mean(y)) * 100
            ),
            "positive_threshold_rate": float(np.mean(prediction > 0)),
        }
    chosen = max(oof_results, key=lambda kind: (
        oof_results[kind]["rank50_advantage_vs_random_points"],
        -oof_results[kind]["mse"],
    ))
    model = _model(chosen, 73)
    model.fit(x, y)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": policy_feature_names(names, "single_reader")}, MODEL_OUT)
    print(f"Chosen on training OOF: {chosen}", flush=True)

    full_report = json.loads((ROOT / "analysis" / "dual_value_full_system.json").read_text())
    judge_roots = [Path(path) for path in full_report["judge_roots"]]
    predictions = _predictions(ROOT / "analysis" / "dual_value_full_system.predictions.jsonl")
    train_ids = {identifier for part in train_parts for identifier in part["ids"]}
    report = {
        "method": "all_reason_holdout_trained_direct_f1_gain_gate_evaluated_on_full_validation_reason_queue",
        "model": chosen,
        "training_questions": len(y),
        "training_counts": {dataset: len(part["ids"])
                            for dataset, part in zip(DATASETS, train_parts, strict=True)},
        "training_label_counts": {
            "positive": int(np.sum(y > 0)),
            "negative": int(np.sum(y < 0)),
            "zero": int(np.sum(y == 0)),
        },
        "oof": oof_results,
        "evaluation": {},
        "notes": [
            "The policy can only skip existing REASON routes; SEARCH and original REASON-queue membership are fixed.",
            "The all-REASON training set and full-validation set are ID-disjoint, but both had been inspected in earlier exploratory work.",
            "Latencies are estimates from cached components, not live end-to-end measurements.",
            "No Qwen inference was run.",
        ],
    }
    for dataset in DATASETS:
        judged = _judge_rows(judge_roots, dataset, "validation")
        data = _selected_features(
            TRAIN_ROOT, dataset, "validation", names, predictions, judged,
        )
        if train_ids.intersection(data["ids"]):
            raise ValueError(f"Training/evaluation ID overlap: {dataset}")
        score = model.predict(data["x"])
        rank = np.argsort(score, kind="stable")
        keep_half = np.zeros(len(score), dtype=bool)
        keep_half[rank[-round(len(score) * 0.5):]] = True
        old = {identifier: predictions[identifier] for identifier in data["ids"]}
        original = full_report["datasets"][dataset]
        count = int(original["count"])
        item = {
            "questions": count,
            "reason_queue_questions": len(data["ids"]),
            "old_full_f1": float(original["f1"] * 100),
            "old_full_em": float(original["em"] * 100),
            "old_full_estimated_latency_seconds": float(original["estimated_latency_seconds"]),
            "old_full_reader_generated_tokens": float(original["average_reader_generated_tokens"]),
            "policies": {},
        }
        for name, keep in (
            ("predicted_positive_gain", score > 0.0),
            ("predicted_gain_over_1_point", score > 0.01),
            ("ranked_keep_half", keep_half),
        ):
            result = _summary(score, data, old, keep)
            multiplier = len(keep) / count
            full_change = result["queue_f1_change_vs_old_points"] * multiplier
            item["policies"][name] = {
                **result,
                "full_f1": float(item["old_full_f1"] + full_change),
                "full_f1_change_vs_old_points": float(full_change),
                "full_f1_change_vs_old_ci95": [
                    float(bound * multiplier) for bound in result["queue_f1_change_vs_old_ci95"]
                ],
                "full_f1_change_vs_matched_random_points": float(
                    result["queue_f1_change_vs_matched_random_points"] * multiplier
                ),
                "full_f1_change_vs_matched_random_ci95": [
                    float(bound * multiplier)
                    for bound in result["queue_f1_change_vs_matched_random_ci95"]
                ],
                "full_em": float(
                    item["old_full_em"]
                    + (result["queue_em"] - data["judged_em"].mean() * 100) * multiplier
                ),
                "full_estimated_latency_seconds": float(
                    item["old_full_estimated_latency_seconds"]
                    - result["saved_latency_seconds_per_queue_question"] * multiplier
                ),
                "full_reader_generated_tokens": float(
                    item["old_full_reader_generated_tokens"]
                    - result["saved_reader_generated_tokens_per_queue_question"] * multiplier
                ),
            }
            print(
                f"{dataset}/{name}: kept {result['reason_kept']}/{len(keep)} REASON, "
                f"F1 {item['policies'][name]['full_f1']:.3f} ({full_change:+.3f} vs old)",
                flush=True,
            )
        report["evaluation"][dataset] = item
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
