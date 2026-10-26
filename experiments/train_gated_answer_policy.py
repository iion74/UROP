"""Train a precision-gated answer-aware one-shot/sequential retrieval policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from experiments.evaluate_answer_oracle import (
    DIRECT_ACTION,
    FIRST_STEP_ACTIONS,
    SECOND_STEP_ACTIONS,
    SEQUENTIAL_STATE_FEATURE_NAMES,
)
from experiments.train_answer_policy import (
    ONE_SHOT_GRAPH_ACTIONS,
    PolicyThresholds,
    _bucket,
    _dataset_weights,
    _fraction,
    _initial_features,
    _load_rows,
    _macro_value,
    _one_shot_targets,
    _report_records,
    _s0_targets,
    _s1_targets,
    _selection_objective,
    _state_features,
    _state_rows,
    _write_predictions,
)
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration
from src.answer_policy import QuestionEvidenceGate
from src.policy import POLICY_FEATURE_NAMES


MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class GatedModels:
    initial_gate: QuestionEvidenceGate
    one_shot_value: ExtraTreesRegressor
    sequential_s0_value: ExtraTreesRegressor
    sequential_s1_gate: QuestionEvidenceGate
    sequential_s1_value: ExtraTreesRegressor


def main() -> None:
    args = _arguments()
    started = perf_counter()
    rows = _load_rows(Path(args.rows_root), args.datasets, args.split, args.max_rows_per_dataset)
    grid = np.linspace(args.gate_threshold_min, args.gate_threshold_max, args.gate_threshold_steps)
    _log(
        f"Gated Answer-aware 정책 학습 시작: {len(rows):,}문항 · "
        f"question TF-IDF + evidence gate · nested folds={args.folds}"
    )
    records, fold_summaries = _nested_audit(rows, args, grid)
    _log("\n[중첩 교차검증: gate·threshold·행동 회귀 모두 test 문항 제외]")
    audit = _report_records(records, args.datasets)
    _log(
        "fold Macro gain (one-shot/sequential): "
        + ", ".join(f"{one:+.3f}/{seq:+.3f}" for one, seq in fold_summaries)
    )

    models, thresholds = _fit_final(rows, args, grid)
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        Path(args.predictions_out)
        if args.predictions_out
        else model_path.with_suffix(".nested_predictions.jsonl")
    )
    _write_predictions(predictions_path, records)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "policy_type": "question_evidence_gated_answer_policy",
            "feature_names": POLICY_FEATURE_NAMES,
            "sequential_state_feature_names": SEQUENTIAL_STATE_FEATURE_NAMES,
            "one_shot_actions": ONE_SHOT_GRAPH_ACTIONS,
            "first_step_actions": FIRST_STEP_ACTIONS,
            "second_step_actions": SECOND_STEP_ACTIONS,
            "thresholds": thresholds.__dict__,
            "initial_gate": models.initial_gate,
            "one_shot_value_model": models.one_shot_value,
            "sequential_s0_value_model": models.sequential_s0_value,
            "sequential_s1_gate": models.sequential_s1_gate,
            "sequential_s1_value_model": models.sequential_s1_value,
        },
        temporary,
    )
    temporary.replace(model_path)
    _write_manifest(
        model_path,
        predictions_path,
        rows,
        args,
        thresholds,
        audit,
        fold_summaries,
    )
    _log(f"\nGated 정책 저장 완료: {model_path}")
    _log(f"중첩 예측 저장 완료: {predictions_path}")
    _log(
        f"최종 gate threshold: one-shot={thresholds.one_shot:.3f}, "
        f"sequential s0/s1={thresholds.sequential_s0:.3f}/{thresholds.sequential_s1:.3f}"
    )
    _log(f"전체 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="질문+근거 gate 기반 Answer-aware 검색 정책")
    parser.add_argument("--rows-root", default="data/benchmarks/multihop/answer_oracle")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--max-features", type=float, default=1.0)
    parser.add_argument("--max-text-features", type=int, default=15_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--gate-threshold-min", type=float, default=0.05)
    parser.add_argument("--gate-threshold-max", type=float, default=0.95)
    parser.add_argument("--gate-threshold-steps", type=int, default=91)
    parser.add_argument("--dataset-harm-penalty", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="data/models/gated_answer_policy.joblib")
    parser.add_argument("--predictions-out")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_rows_per_dataset < 0 or args.folds < 3:
        parser.error("max rows는 0 이상, folds는 3 이상이어야 합니다.")
    if not 0.0 < args.calibration_ratio < 0.5:
        parser.error("calibration ratio는 0과 0.5 사이여야 합니다.")
    if args.trees < 1 or args.min_samples_leaf < 1 or not 0.0 < args.max_features <= 1.0:
        parser.error("tree/leaf/max-features 설정을 확인하세요.")
    if args.max_text_features < 100:
        parser.error("max text features는 100 이상이어야 합니다.")
    if (
        args.gate_threshold_steps < 2
        or not 0.0 <= args.gate_threshold_min < args.gate_threshold_max <= 1.0
    ):
        parser.error("gate threshold 범위를 확인하세요.")
    if args.dataset_harm_penalty < 0:
        parser.error("dataset harm penalty는 0 이상이어야 합니다.")
    return args


def _value_model(args: argparse.Namespace, seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=seed,
        n_jobs=-1,
    )


def _fit_models(rows: list[dict], args: argparse.Namespace, seed_offset: int) -> GatedModels:
    weights = _dataset_weights(rows)
    initial = _initial_features(rows)
    questions = [str(row["question"]) for row in rows]
    graph_labels = np.asarray(
        [
            max(float(row["actions"][action]["f1"]) for action in ONE_SHOT_GRAPH_ACTIONS)
            > float(row["direct_f1"]) + 1e-9
            for row in rows
        ],
        dtype="int64",
    )
    initial_gate = QuestionEvidenceGate.fit(
        initial,
        questions,
        graph_labels,
        sample_weights=weights,
        seed=args.seed + seed_offset,
        max_text_features=args.max_text_features,
    )
    one_value = _value_model(args, args.seed + seed_offset + 1).fit(
        initial,
        _one_shot_targets(rows),
        sample_weight=weights,
    )
    s0_value = _value_model(args, args.seed + seed_offset + 2).fit(
        initial,
        _s0_targets(rows, SECOND_STEP_ACTIONS),
        sample_weight=weights,
    )

    states = _state_rows(rows)
    state_numeric = _state_features(states)
    state_questions = [str(row["question"]) for row, _, _ in states]
    state_weights = np.repeat(weights, len(FIRST_STEP_ACTIONS))
    state_labels = np.asarray(
        [
            max(float(state["action_values"][action]["f1"]) for action in SECOND_STEP_ACTIONS)
            > float(state["action_values"]["stop"]["f1"]) + 1e-9
            for _, _, state in states
        ],
        dtype="int64",
    )
    s1_gate = QuestionEvidenceGate.fit(
        state_numeric,
        state_questions,
        state_labels,
        sample_weights=state_weights,
        seed=args.seed + seed_offset + 3,
        max_text_features=args.max_text_features,
    )
    s1_value = _value_model(args, args.seed + seed_offset + 4).fit(
        state_numeric,
        _s1_targets(states, SECOND_STEP_ACTIONS),
        sample_weight=state_weights,
    )
    return GatedModels(initial_gate, one_value, s0_value, s1_gate, s1_value)


def _model_predictions(models: GatedModels, rows: list[dict]) -> tuple[np.ndarray, ...]:
    initial = _initial_features(rows)
    questions = [str(row["question"]) for row in rows]
    states = _state_rows(rows)
    state_numeric = _state_features(states)
    state_questions = [str(row["question"]) for row, _, _ in states]
    return (
        models.initial_gate.predict_graph_probability(initial, questions),
        models.one_shot_value.predict(initial),
        models.sequential_s0_value.predict(initial),
        models.sequential_s1_gate.predict_graph_probability(state_numeric, state_questions),
        models.sequential_s1_value.predict(state_numeric),
    )


def _nested_audit(
    rows: list[dict],
    args: argparse.Namespace,
    grid: np.ndarray,
) -> tuple[list[dict], list[tuple[float, float]]]:
    records: list[dict] = []
    folds: list[tuple[float, float]] = []
    for fold in range(args.folds):
        test = [row for row in rows if _bucket(row["id"], "outer", args.folds) == fold]
        pool = [row for row in rows if _bucket(row["id"], "outer", args.folds) != fold]
        calibration = [
            row
            for row in pool
            if _fraction(row["id"], f"calibration:{fold}") < args.calibration_ratio
        ]
        calibration_ids = {row["id"] for row in calibration}
        fit = [row for row in pool if row["id"] not in calibration_ids]
        models = _fit_models(fit, args, fold * 10)
        cal_predictions = _model_predictions(models, calibration)
        thresholds = _calibrate(calibration, cal_predictions, args, grid)
        test_predictions = _model_predictions(models, test)
        fold_records = _records(test, test_predictions, thresholds, fold)
        records.extend(fold_records)
        direct = _macro_value(fold_records, "direct_f1", args.datasets)
        one = _macro_value(fold_records, "one_shot_f1", args.datasets)
        seq = _macro_value(fold_records, "sequential_f1", args.datasets)
        folds.append((one - direct, seq - direct))
        _log(
            f"fold {fold + 1}/{args.folds}: fit/cal/test={len(fit)}/{len(calibration)}/{len(test)} "
            f"| Macro gain one-shot/sequential={one - direct:+.3f}/{seq - direct:+.3f} "
            f"| gate={thresholds.one_shot:.2f}/"
            f"{thresholds.sequential_s0:.2f}/{thresholds.sequential_s1:.2f}"
        )
    if len(records) != len(rows) or len({record["id"] for record in records}) != len(rows):
        raise AssertionError("nested OOF 예측이 모든 질문을 정확히 한 번 포함하지 않습니다.")
    return records, folds


def _calibrate(
    rows: list[dict],
    predictions: tuple[np.ndarray, ...],
    args: argparse.Namespace,
    grid: np.ndarray,
) -> PolicyThresholds:
    initial_probability, one_value, s0_value, s1_probability, s1_value = predictions
    one_threshold = max(
        grid,
        key=lambda threshold: (
            _selection_objective(
                rows,
                _one_values(rows, initial_probability, one_value, float(threshold))[0],
                args.datasets,
                args.dataset_harm_penalty,
            ),
            float(threshold),
        ),
    )
    s0_threshold, s1_threshold = max(
        ((float(first), float(second)) for first in grid for second in grid),
        key=lambda thresholds: (
            _selection_objective(
                rows,
                _sequential_values(
                    rows,
                    initial_probability,
                    s0_value,
                    s1_probability,
                    s1_value,
                    *thresholds,
                )[0],
                args.datasets,
                args.dataset_harm_penalty,
            ),
            thresholds[0],
            thresholds[1],
        ),
    )
    return PolicyThresholds(float(one_threshold), s0_threshold, s1_threshold)


def _one_values(
    rows: list[dict],
    gate_probability: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> tuple[list[float], list[str]]:
    selected_values: list[float] = []
    actions: list[str] = []
    for row, probability, predicted_values in zip(rows, gate_probability, values, strict=True):
        index = int(np.argmax(predicted_values))
        action = ONE_SHOT_GRAPH_ACTIONS[index] if float(probability) > threshold else DIRECT_ACTION
        actions.append(action)
        selected_values.append(float(row["actions"][action]["f1"]))
    return selected_values, actions


def _sequential_values(
    rows: list[dict],
    s0_probability: np.ndarray,
    s0_values: np.ndarray,
    s1_probability: np.ndarray,
    s1_values: np.ndarray,
    s0_threshold: float,
    s1_threshold: float,
    *,
    allow_2hop: bool = True,
) -> tuple[list[float], list[str]]:
    selected_values: list[float] = []
    paths: list[str] = []
    for row_index, row in enumerate(rows):
        if float(s0_probability[row_index]) <= s0_threshold:
            path = "stop"
        else:
            first_index = int(np.argmax(s0_values[row_index]))
            first_action = FIRST_STEP_ACTIONS[first_index]
            state_index = row_index * len(FIRST_STEP_ACTIONS) + first_index
            if allow_2hop and float(s1_probability[state_index]) > s1_threshold:
                second_index = int(np.argmax(s1_values[state_index]))
                second_action = SECOND_STEP_ACTIONS[second_index]
            else:
                second_action = "stop"
            path = row["sequential"]["states_after_first_step"][first_action][
                "action_trajectories"
            ][second_action]
        selected_values.append(float(row["sequential"]["terminal_results"][path]["f1"]))
        paths.append(path)
    return selected_values, paths


def _records(
    rows: list[dict],
    predictions: tuple[np.ndarray, ...],
    thresholds: PolicyThresholds,
    fold: int,
) -> list[dict]:
    initial_probability, one_value, s0_value, s1_probability, s1_value = predictions
    one_values, actions = _one_values(
        rows, initial_probability, one_value, thresholds.one_shot
    )
    sequential_values, paths = _sequential_values(
        rows,
        initial_probability,
        s0_value,
        s1_probability,
        s1_value,
        thresholds.sequential_s0,
        thresholds.sequential_s1,
    )
    no_2hop_values, _ = _sequential_values(
        rows,
        initial_probability,
        s0_value,
        s1_probability,
        s1_value,
        thresholds.sequential_s0,
        thresholds.sequential_s1,
        allow_2hop=False,
    )
    return [
        {
            "id": row["id"],
            "dataset": row["dataset"],
            "fold": fold,
            "direct_f1": float(row["direct_f1"]),
            "one_shot_f1": float(one_f1),
            "one_shot_action": action,
            "sequential_f1": float(seq_f1),
            "sequential_no_2hop_f1": float(no_2hop_f1),
            "sequential_path": path,
            "oracle_f1": float(row["oracle_f1"]),
        }
        for row, one_f1, action, seq_f1, no_2hop_f1, path in zip(
            rows,
            one_values,
            actions,
            sequential_values,
            no_2hop_values,
            paths,
            strict=True,
        )
    ]


def _fit_final(
    rows: list[dict],
    args: argparse.Namespace,
    grid: np.ndarray,
) -> tuple[GatedModels, PolicyThresholds]:
    initial_probability = np.empty(len(rows), dtype="float64")
    one_value = np.empty((len(rows), len(ONE_SHOT_GRAPH_ACTIONS)), dtype="float64")
    s0_value = np.empty((len(rows), len(FIRST_STEP_ACTIONS)), dtype="float64")
    state_count = len(rows) * len(FIRST_STEP_ACTIONS)
    s1_probability = np.empty(state_count, dtype="float64")
    s1_value = np.empty((state_count, len(SECOND_STEP_ACTIONS)), dtype="float64")
    for fold in range(args.folds):
        indices = [
            index
            for index, row in enumerate(rows)
            if _bucket(row["id"], "final-oof", args.folds) == fold
        ]
        fit = [
            row
            for row in rows
            if _bucket(row["id"], "final-oof", args.folds) != fold
        ]
        test = [rows[index] for index in indices]
        predictions = _model_predictions(_fit_models(fit, args, 10_000 + fold * 10), test)
        initial_probability[indices] = predictions[0]
        one_value[indices] = predictions[1]
        s0_value[indices] = predictions[2]
        for local, global_index in enumerate(indices):
            local_start = local * len(FIRST_STEP_ACTIONS)
            global_start = global_index * len(FIRST_STEP_ACTIONS)
            s1_probability[global_start:global_start + len(FIRST_STEP_ACTIONS)] = (
                predictions[3][local_start:local_start + len(FIRST_STEP_ACTIONS)]
            )
            s1_value[global_start:global_start + len(FIRST_STEP_ACTIONS)] = (
                predictions[4][local_start:local_start + len(FIRST_STEP_ACTIONS)]
            )
    thresholds = _calibrate(
        rows,
        (initial_probability, one_value, s0_value, s1_probability, s1_value),
        args,
        grid,
    )
    return _fit_models(rows, args, 20_000), thresholds


def _write_manifest(
    model_path: Path,
    predictions_path: Path,
    rows: list[dict],
    args: argparse.Namespace,
    thresholds: PolicyThresholds,
    audit: dict,
    fold_summaries: list[tuple[float, float]],
) -> None:
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "policy_type": "question_evidence_gated_answer_policy",
        "model": str(model_path),
        "nested_predictions": str(predictions_path),
        "datasets": list(args.datasets),
        "rows": len(rows),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "seed": args.seed,
        "folds": args.folds,
        "calibration_ratio": args.calibration_ratio,
        "final_threshold_calibration": f"{args.folds}-fold-oof",
        "trees": args.trees,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "max_text_features": args.max_text_features,
        "dataset_harm_penalty": args.dataset_harm_penalty,
        "thresholds": thresholds.__dict__,
        "nested_audit": audit,
        "fold_macro_gains": [
            {"one_shot": one, "sequential": seq} for one, seq in fold_summaries
        ],
        "row_fingerprint": hashlib.sha256(
            "\n".join(sorted(str(row["id"]) for row in rows)).encode("utf-8")
        ).hexdigest(),
    }
    path = model_path.with_suffix(".manifest.json")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
