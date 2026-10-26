"""Train and audit one-shot versus sequential answer-aware retrieval policies.

Only retrieval routing is learned.  The retriever, graph action executor and QA
reader outputs are frozen in the Sequential v5 JSONL rows.  Nested development
folds estimate policy F1 without evaluating a row with a model fitted on that
row; final models are then fitted on all train rows for later validation use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from experiments.evaluate_answer_oracle import (
    DIRECT_ACTION,
    FIRST_STEP_ACTIONS,
    POST_FIRST_STEP_ACTIONS,
    SECOND_STEP_ACTIONS,
    SEQUENTIAL_OUTPUT_FORMAT_VERSION,
    SEQUENTIAL_STATE_FEATURE_NAMES,
)
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from src.policy import POLICY_FEATURE_NAMES
from src.routing import POLICY_ACTION_IDS


ONE_SHOT_GRAPH_ACTIONS = tuple(action for action in POLICY_ACTION_IDS if action != DIRECT_ACTION)
MODEL_FORMAT_VERSION = 2


@dataclass(frozen=True)
class PolicyModels:
    one_shot: object
    sequential_s0: object
    sequential_s1: object


@dataclass(frozen=True)
class PolicyThresholds:
    one_shot: float
    sequential_s0: float
    sequential_s1: float


def main() -> None:
    args = _arguments()
    started = perf_counter()
    rows = _load_rows(Path(args.rows_root), args.datasets, args.split, args.max_rows_per_dataset)
    _log(
        f"Answer-aware 검색 정책 학습 시작: {len(rows):,}문항 · "
        f"model={args.model_type} · nested folds={args.folds} · "
        f"s1 actions={_post_actions(args)}"
    )
    _log("QA reader와 검색 결과는 고정되며 one-shot/sequential 정책만 학습합니다.")

    grid = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    nested_records, fold_summaries = _nested_audit(rows, args, grid)
    _log("\n[중첩 교차검증: 모든 평가는 해당 문항을 보지 않은 정책으로 계산]")
    audit = _report_records(nested_records, args.datasets)
    _log(
        "fold Macro gain (one-shot/sequential): "
        + ", ".join(f"{one:+.3f}/{seq:+.3f}" for one, seq in fold_summaries)
    )

    final_models, final_thresholds = _fit_final(rows, args, grid)
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        Path(args.predictions_out)
        if args.predictions_out
        else model_path.with_suffix(".nested_predictions.jsonl")
    )
    _write_predictions(predictions_path, nested_records)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "model_type": args.model_type,
            "feature_names": POLICY_FEATURE_NAMES,
            "sequential_state_feature_names": SEQUENTIAL_STATE_FEATURE_NAMES,
            "one_shot_actions": ONE_SHOT_GRAPH_ACTIONS,
            "first_step_actions": FIRST_STEP_ACTIONS,
            "post_first_step_actions": _post_actions(args),
            "thresholds": final_thresholds.__dict__,
            "one_shot_model": final_models.one_shot,
            "sequential_s0_model": final_models.sequential_s0,
            "sequential_s1_model": final_models.sequential_s1,
        },
        temporary,
    )
    temporary.replace(model_path)
    _write_manifest(
        model_path,
        args,
        rows,
        final_thresholds,
        audit,
        fold_summaries,
        predictions_path=predictions_path,
    )
    _log(f"\n정책 저장 완료: {model_path}")
    _log(f"중첩 예측 저장 완료: {predictions_path}")
    _log(
        f"최종 threshold: one-shot={final_thresholds.one_shot:+.3f}, "
        f"sequential s0/s1={final_thresholds.sequential_s0:+.3f}/"
        f"{final_thresholds.sequential_s1:+.3f}"
    )
    _log(f"전체 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer F1 기반 one-shot/sequential 검색 정책 학습")
    parser.add_argument("--rows-root", default="data/benchmarks/multihop/answer_oracle")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument("--model-type", choices=("extra_trees", "random_forest", "ridge"), default="extra_trees")
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--max-features", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--threshold-min", type=float, default=-0.08)
    parser.add_argument("--threshold-max", type=float, default=1.00)
    parser.add_argument("--threshold-steps", type=int, default=109)
    parser.add_argument("--dataset-harm-penalty", type=float, default=2.0)
    parser.add_argument(
        "--enable-return-direct",
        action="store_true",
        help="1-hop 뒤 Direct evidence 복귀 행동을 포함하는 ablation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="data/models/answer_aware_sequential_policy.joblib")
    parser.add_argument("--predictions-out")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_rows_per_dataset < 0:
        parser.error("max rows는 0 이상이어야 합니다.")
    if args.folds < 3 or not 0.0 < args.calibration_ratio < 0.5:
        parser.error("folds는 3 이상, calibration ratio는 0과 0.5 사이여야 합니다.")
    if args.trees < 1 or args.min_samples_leaf < 1 or not 0.0 < args.max_features <= 1.0:
        parser.error("tree 수/leaf는 1 이상, max-features는 0과 1 사이여야 합니다.")
    if args.threshold_steps < 2 or args.threshold_min >= args.threshold_max:
        parser.error("threshold 범위를 확인하세요.")
    if args.dataset_harm_penalty < 0:
        parser.error("dataset harm penalty는 0 이상이어야 합니다.")
    return args


def _load_rows(root: Path, datasets: tuple[str, ...], split: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    for dataset in datasets:
        path = root / f"{dataset}_{split}_sequential_v6.jsonl"
        dataset_rows = list(_iter_jsonl(path))
        if limit:
            dataset_rows = dataset_rows[:limit]
        if not dataset_rows:
            raise SystemExit(f"Sequential v6 상태가 없습니다: {path}")
        for row in dataset_rows:
            if int(row.get("format_version", -1)) != SEQUENTIAL_OUTPUT_FORMAT_VERSION:
                raise SystemExit(f"지원하지 않는 Sequential 행 형식입니다: {path}")
            if row.get("dataset") != dataset or not row.get("sequential", {}).get(
                "shared_terminal_oracle"
            ):
                raise SystemExit(f"공유 terminal Oracle 행이 아닙니다: {row.get('id')}")
            if not (
                np.isclose(float(row["oracle_f1"]), float(row["sequential_oracle_f1"]))
                and np.isclose(float(row["oracle_em"]), float(row["sequential_oracle_em"]))
            ):
                raise SystemExit(f"one-shot/sequential Oracle 불일치: {row.get('id')}")
        rows.extend(dataset_rows)
    if len(rows) < len(datasets) * 10:
        raise SystemExit("정책 학습/교차검증에 필요한 행이 너무 적습니다.")
    return rows


def _make_model(args: argparse.Namespace, seed: int):
    if args.model_type == "ridge":
        return Ridge(alpha=args.ridge_alpha)
    common = {
        "n_estimators": args.trees,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "random_state": seed,
        "n_jobs": -1,
    }
    if args.model_type == "random_forest":
        return RandomForestRegressor(**common)
    return ExtraTreesRegressor(**common)


def _fit_models(rows: list[dict], args: argparse.Namespace, seed_offset: int) -> PolicyModels:
    sample_weights = _dataset_weights(rows)
    states = _state_rows(rows)
    state_weights = np.repeat(sample_weights, len(FIRST_STEP_ACTIONS))
    one_shot = _make_model(args, args.seed + seed_offset).fit(
        _initial_features(rows),
        _one_shot_targets(rows),
        sample_weight=sample_weights,
    )
    sequential_s0 = _make_model(args, args.seed + seed_offset + 1).fit(
        _initial_features(rows),
        _s0_targets(rows, _post_actions(args)),
        sample_weight=sample_weights,
    )
    sequential_s1 = _make_model(args, args.seed + seed_offset + 2).fit(
        _state_features(states),
        _s1_targets(states, _post_actions(args)),
        sample_weight=state_weights,
    )
    return PolicyModels(one_shot, sequential_s0, sequential_s1)


def _nested_audit(
    rows: list[dict],
    args: argparse.Namespace,
    grid: np.ndarray,
) -> tuple[list[dict], list[tuple[float, float]]]:
    records: list[dict] = []
    fold_summaries: list[tuple[float, float]] = []
    for fold in range(args.folds):
        test = [row for row in rows if _bucket(row["id"], "outer", args.folds) == fold]
        pool = [row for row in rows if _bucket(row["id"], "outer", args.folds) != fold]
        calibration = [
            row
            for row in pool
            if _fraction(row["id"], f"calibration:{fold}") < args.calibration_ratio
        ]
        fit = [row for row in pool if row not in calibration]
        if not fit or not calibration or not test:
            raise ValueError(f"fold {fold}의 fit/calibration/test 중 하나가 비었습니다.")
        models = _fit_models(fit, args, seed_offset=fold * 10)
        thresholds = _calibrate(models, calibration, args, grid)
        fold_records = _predict_records(models, thresholds, test, args=args, fold=fold)
        records.extend(fold_records)
        direct = _macro_value(fold_records, "direct_f1", args.datasets)
        one_shot = _macro_value(fold_records, "one_shot_f1", args.datasets)
        sequential = _macro_value(fold_records, "sequential_f1", args.datasets)
        fold_summaries.append((one_shot - direct, sequential - direct))
        _log(
            f"fold {fold + 1}/{args.folds}: fit/cal/test={len(fit)}/{len(calibration)}/{len(test)} "
            f"| Macro gain one-shot/sequential={one_shot - direct:+.3f}/{sequential - direct:+.3f} "
            f"| threshold={thresholds.one_shot:+.2f}/"
            f"{thresholds.sequential_s0:+.2f}/{thresholds.sequential_s1:+.2f}"
        )
    if len(records) != len(rows) or len({record["id"] for record in records}) != len(rows):
        raise AssertionError("중첩 교차검증 예측이 모든 질문을 정확히 한 번 포함하지 않습니다.")
    return records, fold_summaries


def _fit_final(
    rows: list[dict],
    args: argparse.Namespace,
    grid: np.ndarray,
) -> tuple[PolicyModels, PolicyThresholds]:
    one_predictions = np.empty((len(rows), len(ONE_SHOT_GRAPH_ACTIONS)), dtype="float64")
    s0_predictions = np.empty((len(rows), len(FIRST_STEP_ACTIONS)), dtype="float64")
    s1_predictions = np.empty(
        (len(rows) * len(FIRST_STEP_ACTIONS), len(_post_actions(args))), dtype="float64"
    )
    for fold in range(args.folds):
        test_indices = [
            index
            for index, row in enumerate(rows)
            if _bucket(row["id"], "final-oof", args.folds) == fold
        ]
        fit = [
            row
            for row in rows
            if _bucket(row["id"], "final-oof", args.folds) != fold
        ]
        test = [rows[index] for index in test_indices]
        models = _fit_models(fit, args, seed_offset=10_000 + fold * 10)
        one_predictions[test_indices] = models.one_shot.predict(_initial_features(test))
        s0_predictions[test_indices] = models.sequential_s0.predict(_initial_features(test))
        state_prediction = models.sequential_s1.predict(_state_features(_state_rows(test)))
        for local_index, global_index in enumerate(test_indices):
            start = local_index * len(FIRST_STEP_ACTIONS)
            global_start = global_index * len(FIRST_STEP_ACTIONS)
            s1_predictions[global_start:global_start + len(FIRST_STEP_ACTIONS)] = (
                state_prediction[start:start + len(FIRST_STEP_ACTIONS)]
            )
    thresholds = _calibrate_predictions(
        rows,
        one_predictions,
        s0_predictions,
        s1_predictions,
        args,
        grid,
    )
    final_models = _fit_models(rows, args, seed_offset=20_000)
    return final_models, thresholds


def _calibrate(
    models: PolicyModels,
    rows: list[dict],
    args: argparse.Namespace,
    grid: np.ndarray,
) -> PolicyThresholds:
    one_predictions = models.one_shot.predict(_initial_features(rows))
    s0_predictions = models.sequential_s0.predict(_initial_features(rows))
    s1_predictions = models.sequential_s1.predict(_state_features(_state_rows(rows)))
    return _calibrate_predictions(
        rows,
        one_predictions,
        s0_predictions,
        s1_predictions,
        args,
        grid,
    )


def _calibrate_predictions(
    rows: list[dict],
    one_predictions: np.ndarray,
    s0_predictions: np.ndarray,
    s1_predictions: np.ndarray,
    args: argparse.Namespace,
    grid: np.ndarray,
) -> PolicyThresholds:
    one_threshold = max(
        grid,
        key=lambda threshold: (
            _selection_objective(
                rows,
                _one_shot_values(rows, one_predictions, float(threshold)),
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
                    s0_predictions,
                    s1_predictions,
                    *thresholds,
                    post_actions=_post_actions(args),
                )[0],
                args.datasets,
                args.dataset_harm_penalty,
            ),
            thresholds[0],
            thresholds[1],
        ),
    )
    return PolicyThresholds(float(one_threshold), s0_threshold, s1_threshold)


def _predict_records(
    models: PolicyModels,
    thresholds: PolicyThresholds,
    rows: list[dict],
    *,
    args: argparse.Namespace,
    fold: int,
) -> list[dict]:
    one_predictions = models.one_shot.predict(_initial_features(rows))
    one_values, one_actions = _one_shot_values(
        rows, one_predictions, thresholds.one_shot, return_actions=True
    )
    s0_predictions = models.sequential_s0.predict(_initial_features(rows))
    s1_predictions = models.sequential_s1.predict(_state_features(_state_rows(rows)))
    sequential_values, paths = _sequential_values(
        rows,
        s0_predictions,
        s1_predictions,
        thresholds.sequential_s0,
        thresholds.sequential_s1,
        post_actions=_post_actions(args),
    )
    no_2hop_values, _ = _sequential_values(
        rows,
        s0_predictions,
        s1_predictions,
        thresholds.sequential_s0,
        thresholds.sequential_s1,
        allow_2hop=False,
        post_actions=_post_actions(args),
    )
    return [
        {
            "id": row["id"],
            "dataset": row["dataset"],
            "fold": fold,
            "direct_f1": float(row["direct_f1"]),
            "one_shot_f1": float(one_f1),
            "one_shot_action": action,
            "sequential_f1": float(sequential_f1),
            "sequential_no_2hop_f1": float(no_2hop_f1),
            "sequential_path": path,
            "oracle_f1": float(row["oracle_f1"]),
        }
        for row, one_f1, action, sequential_f1, no_2hop_f1, path in zip(
            rows,
            one_values,
            one_actions,
            sequential_values,
            no_2hop_values,
            paths,
            strict=True,
        )
    ]


def _one_shot_values(
    rows: list[dict],
    predictions: np.ndarray,
    threshold: float,
    *,
    return_actions: bool = False,
):
    actions: list[str] = []
    values: list[float] = []
    for row, prediction in zip(rows, predictions, strict=True):
        index = int(np.argmax(prediction))
        action = ONE_SHOT_GRAPH_ACTIONS[index] if float(prediction[index]) > threshold else DIRECT_ACTION
        actions.append(action)
        values.append(float(row["actions"][action]["f1"]))
    return (values, actions) if return_actions else values


def _sequential_values(
    rows: list[dict],
    s0_predictions: np.ndarray,
    s1_predictions: np.ndarray,
    s0_threshold: float,
    s1_threshold: float,
    *,
    allow_2hop: bool = True,
    post_actions: tuple[str, ...] = SECOND_STEP_ACTIONS,
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    paths: list[str] = []
    for row_index, (row, s0_prediction) in enumerate(zip(rows, s0_predictions, strict=True)):
        first_index = int(np.argmax(s0_prediction))
        if float(s0_prediction[first_index]) <= s0_threshold:
            path = "stop"
        else:
            first_action = FIRST_STEP_ACTIONS[first_index]
            state_prediction = np.array(
                s1_predictions[row_index * len(FIRST_STEP_ACTIONS) + first_index], copy=True
            )
            if not allow_2hop:
                for index, action in enumerate(post_actions):
                    if action in SECOND_STEP_ACTIONS:
                        state_prediction[index] = -np.inf
            second_index = int(np.argmax(state_prediction))
            second_action = (
                post_actions[second_index]
                if float(state_prediction[second_index]) > s1_threshold
                else "stop"
            )
            path = row["sequential"]["states_after_first_step"][first_action][
                "action_trajectories"
            ][second_action]
        values.append(float(row["sequential"]["terminal_results"][path]["f1"]))
        paths.append(path)
    return values, paths


def _initial_features(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in POLICY_FEATURE_NAMES] for row in rows],
        dtype="float64",
    )


def _one_shot_targets(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            [float(row["actions"][action]["f1"]) - float(row["direct_f1"]) for action in ONE_SHOT_GRAPH_ACTIONS]
            for row in rows
        ],
        dtype="float64",
    )


def _s0_targets(
    rows: list[dict],
    post_actions: tuple[str, ...] = SECOND_STEP_ACTIONS,
) -> np.ndarray:
    return np.asarray(
        [
            [
                max(
                    float(
                        row["sequential"]["states_after_first_step"][action]["action_values"][candidate]["f1"]
                    )
                    for candidate in ("stop", *post_actions)
                )
                - float(row["direct_f1"])
                for action in FIRST_STEP_ACTIONS
            ]
            for row in rows
        ],
        dtype="float64",
    )


def _state_rows(rows: list[dict]) -> list[tuple[dict, str, dict]]:
    return [
        (row, action, row["sequential"]["states_after_first_step"][action])
        for row in rows
        for action in FIRST_STEP_ACTIONS
    ]


def _state_features(states: list[tuple[dict, str, dict]]) -> np.ndarray:
    return np.asarray(
        [[float(state["features"][name]) for name in SEQUENTIAL_STATE_FEATURE_NAMES] for _, _, state in states],
        dtype="float64",
    )


def _s1_targets(
    states: list[tuple[dict, str, dict]],
    post_actions: tuple[str, ...] = SECOND_STEP_ACTIONS,
) -> np.ndarray:
    return np.asarray(
        [
            [
                float(state["action_values"][action]["f1"])
                - float(state["action_values"]["stop"]["f1"])
                for action in post_actions
            ]
            for _, _, state in states
        ],
        dtype="float64",
    )


def _dataset_weights(rows: list[dict]) -> np.ndarray:
    counts = Counter(str(row["dataset"]) for row in rows)
    return np.asarray(
        [len(rows) / (len(counts) * counts[str(row["dataset"])]) for row in rows],
        dtype="float64",
    )


def _selection_objective(
    rows: list[dict],
    values: list[float],
    datasets: Iterable[str],
    harm_penalty: float,
) -> float:
    policy = _dataset_values(rows, values, datasets)
    direct = _dataset_values(rows, [float(row["direct_f1"]) for row in rows], datasets)
    return float(np.mean(policy) - harm_penalty * np.maximum(0.0, direct - policy).sum())


def _dataset_values(rows: list[dict], values: list[float], datasets: Iterable[str]) -> np.ndarray:
    return np.asarray(
        [
            np.mean([value for row, value in zip(rows, values, strict=True) if row["dataset"] == dataset])
            for dataset in datasets
        ],
        dtype="float64",
    )


def _macro_value(records: list[dict], key: str, datasets: Iterable[str]) -> float:
    return float(
        np.mean(
            [np.mean([record[key] for record in records if record["dataset"] == dataset]) for dataset in datasets]
        )
    )


def _report_records(records: list[dict], datasets: Iterable[str]) -> dict:
    summary = {}
    for name in (*datasets, "Macro"):
        selected = records if name == "Macro" else [record for record in records if record["dataset"] == name]
        if name == "Macro":
            direct = _macro_value(records, "direct_f1", datasets)
            one_shot = _macro_value(records, "one_shot_f1", datasets)
            sequential = _macro_value(records, "sequential_f1", datasets)
            no_2hop = _macro_value(records, "sequential_no_2hop_f1", datasets)
            oracle = _macro_value(records, "oracle_f1", datasets)
        else:
            direct = float(np.mean([record["direct_f1"] for record in selected]))
            one_shot = float(np.mean([record["one_shot_f1"] for record in selected]))
            sequential = float(np.mean([record["sequential_f1"] for record in selected]))
            no_2hop = float(np.mean([record["sequential_no_2hop_f1"] for record in selected]))
            oracle = float(np.mean([record["oracle_f1"] for record in selected]))
        potential = oracle - direct
        capture = (sequential - direct) / potential if potential > 1e-12 else 0.0
        _log(
            f"{name}: n={len(selected):,} | F1 direct/one-shot/sequential/oracle="
            f"{direct:.3f}/{one_shot:.3f}/{sequential:.3f}/{oracle:.3f} "
            f"| gain one-shot/sequential={one_shot - direct:+.3f}/{sequential - direct:+.3f} "
            f"| sequential-one-shot={sequential - one_shot:+.3f} "
            f"| 2-hop decision 기여={sequential - no_2hop:+.3f} "
            f"| regret={oracle - sequential:.3f} | capture={capture:.1%}"
        )
        summary[name] = {
            "n": len(selected),
            "direct_f1": direct,
            "one_shot_f1": one_shot,
            "sequential_f1": sequential,
            "oracle_f1": oracle,
            "one_shot_gain": one_shot - direct,
            "sequential_gain": sequential - direct,
            "sequential_vs_one_shot": sequential - one_shot,
            "sequential_no_2hop_f1": no_2hop,
            "second_hop_contribution": sequential - no_2hop,
            "sequential_regret": oracle - sequential,
            "capture_ratio": capture,
        }
    _log(f"one-shot 행동={dict(sorted(Counter(record['one_shot_action'] for record in records).items()))}")
    _log(f"sequential 경로={dict(sorted(Counter(record['sequential_path'] for record in records).items()))}")
    return summary


def _write_manifest(
    model_path: Path,
    args: argparse.Namespace,
    rows: list[dict],
    thresholds: PolicyThresholds,
    audit: dict,
    fold_summaries: list[tuple[float, float]],
    *,
    predictions_path: Path,
) -> None:
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "model": str(model_path),
        "nested_predictions": str(predictions_path),
        "model_type": args.model_type,
        "datasets": list(args.datasets),
        "split": args.split,
        "rows": len(rows),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "seed": args.seed,
        "folds": args.folds,
        "calibration_ratio": args.calibration_ratio,
        "final_threshold_calibration": f"{args.folds}-fold-oof",
        "trees": args.trees if args.model_type != "ridge" else None,
        "min_samples_leaf": args.min_samples_leaf if args.model_type != "ridge" else None,
        "max_features": args.max_features if args.model_type != "ridge" else None,
        "ridge_alpha": args.ridge_alpha if args.model_type == "ridge" else None,
        "dataset_harm_penalty": args.dataset_harm_penalty,
        "enable_return_direct": args.enable_return_direct,
        "post_first_step_actions": list(_post_actions(args)),
        "thresholds": thresholds.__dict__,
        "nested_audit": audit,
        "fold_macro_gains": [
            {"one_shot": one_shot, "sequential": sequential}
            for one_shot, sequential in fold_summaries
        ],
        "row_fingerprint": hashlib.sha256(
            "\n".join(sorted(str(row["id"]) for row in rows)).encode("utf-8")
        ).hexdigest(),
    }
    manifest = model_path.with_suffix(".manifest.json")
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest)


def _write_predictions(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for record in sorted(records, key=lambda item: (item["dataset"], item["id"])):
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _bucket(identifier: str, salt: str, buckets: int) -> int:
    return int(hashlib.sha256(f"{salt}:{identifier}".encode("utf-8")).hexdigest()[:8], 16) % buckets


def _fraction(identifier: str, salt: str) -> float:
    return int(hashlib.sha256(f"{salt}:{identifier}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _post_actions(args: argparse.Namespace) -> tuple[str, ...]:
    return POST_FIRST_STEP_ACTIONS if args.enable_return_direct else SECOND_STEP_ACTIONS


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
