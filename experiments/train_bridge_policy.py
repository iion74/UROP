"""Train a pilot policy over Direct, legacy Graph, and Bridge-path actions.

The trainer merges the completed legacy Answer Oracle rows with the new
evidence-conditioned Bridge Answer rows.  Nested folds keep every test question
out of gate fitting, value fitting, and threshold calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from experiments.evaluate_answer_oracle import DIRECT_ACTION
from experiments.train_answer_policy import _bucket, _fraction, _macro_value, _selection_objective
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from src.answer_policy import QuestionEvidenceGate
from src.policy import POLICY_FEATURE_NAMES


FORMAT_VERSION = 1
LEGACY_ACTIONS = (
    "graph_1hop_small",
    "graph_1hop_large",
    "graph_2hop_small",
    "graph_2hop_large",
)
NEW_ACTIONS = ("bridge_path_1", "bridge_path_2")
NON_DIRECT_ACTIONS = (*LEGACY_ACTIONS, *NEW_ACTIONS)
ALL_ACTIONS = (DIRECT_ACTION, *NON_DIRECT_ACTIONS)
BRIDGE_FEATURE_NAMES = (
    "bridge_evaluated_pairs",
    "bridge_distinct_candidates",
    "bridge_top_path_score",
    "bridge_top_path_margin",
    "bridge_top_path_title_mentioned",
)
FEATURE_NAMES = (*POLICY_FEATURE_NAMES, *BRIDGE_FEATURE_NAMES)


@dataclass(frozen=True)
class BridgePolicyModels:
    gate: QuestionEvidenceGate
    value: ExtraTreesRegressor


@dataclass(frozen=True)
class Thresholds:
    gate: float
    advantage: float


def main() -> None:
    args = _arguments()
    started = perf_counter()
    rows = _load_rows(
        Path(args.base_oracle_root),
        Path(args.bridge_oracle_root),
        args.datasets,
        args.split,
        args.max_rows_per_dataset,
    )
    positives = sum(
        max(float(row["actions"][action]["f1"]) for action in NON_DIRECT_ACTIONS)
        > float(row["direct_f1"]) + 1e-9
        for row in rows
    )
    _log(
        f"Bridge 정책 pilot 학습 시작: {len(rows):,}문항 · beneficial={positives:,} "
        f"({positives / len(rows):.1%}) · nested folds={args.folds}"
    )
    _report_static(rows, args.datasets)

    gate_grid = np.linspace(args.gate_threshold_min, args.gate_threshold_max, args.gate_threshold_steps)
    advantage_grid = np.linspace(
        args.advantage_threshold_min,
        args.advantage_threshold_max,
        args.advantage_threshold_steps,
    )
    records, fold_gains = _nested_audit(rows, args, gate_grid, advantage_grid)
    audit = _report_records(records, args.datasets)
    models, thresholds = _fit_final(rows, args, gate_grid, advantage_grid)

    model_path = Path(args.model_out)
    predictions_path = (
        Path(args.predictions_out)
        if args.predictions_out
        else model_path.with_suffix(".nested_predictions.jsonl")
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(
        {
            "format_version": FORMAT_VERSION,
            "policy_type": "evidence_conditioned_bridge_gated_policy",
            "feature_names": FEATURE_NAMES,
            "non_direct_actions": NON_DIRECT_ACTIONS,
            "thresholds": {"gate": thresholds.gate, "advantage": thresholds.advantage},
            "gate": models.gate,
            "value_model": models.value,
        },
        temporary,
    )
    temporary.replace(model_path)
    _write_predictions(predictions_path, records)
    manifest = {
        "format_version": FORMAT_VERSION,
        "policy_type": "evidence_conditioned_bridge_gated_policy",
        "model": str(model_path),
        "nested_predictions": str(predictions_path),
        "datasets": list(args.datasets),
        "split": args.split,
        "rows": len(rows),
        "dataset_counts": dict(sorted(Counter(str(row["dataset"]) for row in rows).items())),
        "feature_names": list(FEATURE_NAMES),
        "actions": list(ALL_ACTIONS),
        "target": "answer_f1_advantage_over_direct",
        "seed": args.seed,
        "folds": args.folds,
        "thresholds": {"gate": thresholds.gate, "advantage": thresholds.advantage},
        "nested_audit": audit,
        "fold_macro_gains": fold_gains,
        "row_fingerprint": hashlib.sha256(
            "\n".join(sorted(str(row["id"]) for row in rows)).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path = model_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"정책 저장: {model_path}")
    _log(f"최종 threshold gate/advantage={thresholds.gate:.3f}/{thresholds.advantage:+.3f}")
    _log(f"전체 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct/Graph/Bridge Answer-F1 정책 pilot")
    parser.add_argument("--base-oracle-root", default="data/benchmarks/multihop/answer_oracle")
    parser.add_argument("--bridge-oracle-root", default="data/benchmarks/multihop/bridge_answer_oracle_v1")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=1.0)
    parser.add_argument("--max-text-features", type=int, default=15_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--gate-threshold-min", type=float, default=0.10)
    parser.add_argument("--gate-threshold-max", type=float, default=0.90)
    parser.add_argument("--gate-threshold-steps", type=int, default=17)
    parser.add_argument("--advantage-threshold-min", type=float, default=-0.05)
    parser.add_argument("--advantage-threshold-max", type=float, default=0.15)
    parser.add_argument("--advantage-threshold-steps", type=int, default=21)
    parser.add_argument("--dataset-harm-penalty", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="data/models/bridge_policy_pilot_100.joblib")
    parser.add_argument("--predictions-out")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_rows_per_dataset < 0 or args.folds < 3:
        parser.error("max rows는 0 이상, folds는 3 이상이어야 합니다.")
    if not 0.0 < args.calibration_ratio < 0.5:
        parser.error("calibration ratio는 0과 0.5 사이여야 합니다.")
    if args.trees < 1 or args.min_samples_leaf < 1 or not 0.0 < args.max_features <= 1.0:
        parser.error("tree/leaf/max-features 설정을 확인하세요.")
    if args.max_text_features < 100 or args.dataset_harm_penalty < 0.0:
        parser.error("text feature 수와 harm penalty를 확인하세요.")
    if args.gate_threshold_steps < 2 or args.advantage_threshold_steps < 2:
        parser.error("threshold grid 크기는 2 이상이어야 합니다.")
    return args


def _load_rows(
    base_root: Path,
    bridge_root: Path,
    datasets: tuple[str, ...],
    split: str,
    maximum: int,
) -> list[dict]:
    rows: list[dict] = []
    for dataset in datasets:
        base_path = base_root / f"{dataset}_{split}.jsonl"
        bridge_path = bridge_root / f"{dataset}_{split}.jsonl"
        if not base_path.exists() or not bridge_path.exists():
            raise SystemExit(f"Oracle 결과가 없습니다: {base_path} / {bridge_path}")
        bridge_rows = list(_iter_jsonl(bridge_path))
        if maximum:
            bridge_rows = bridge_rows[:maximum]
        bridge_by_id = {str(row["id"]): row for row in bridge_rows}
        base_by_id = {
            str(row["id"]): row
            for row in _iter_jsonl(base_path)
            if str(row.get("id")) in bridge_by_id
        }
        if set(base_by_id) != set(bridge_by_id):
            raise SystemExit(f"{dataset}: base/bridge Oracle ID가 일치하지 않습니다.")
        for row_id, bridge in bridge_by_id.items():
            base = base_by_id[row_id]
            actions = {
                action: {
                    "em": float(base["actions"][action]["em"]),
                    "f1": float(base["actions"][action]["f1"]),
                }
                for action in (DIRECT_ACTION, *LEGACY_ACTIONS)
            }
            actions.update(
                {
                    action: {
                        "em": float(bridge["new_actions"][action]["em"]),
                        "f1": float(bridge["new_actions"][action]["f1"]),
                    }
                    for action in NEW_ACTIONS
                }
            )
            audit = bridge["bridge_audit"]
            features = {name: float(base["features"][name]) for name in POLICY_FEATURE_NAMES}
            features.update(
                {
                    "bridge_evaluated_pairs": min(1.0, float(audit["evaluated_pairs"]) / 48.0),
                    "bridge_distinct_candidates": min(1.0, float(audit["distinct_candidates"]) / 48.0),
                    "bridge_top_path_score": float(audit["top_path_score"]),
                    "bridge_top_path_margin": float(audit["top_path_margin"]),
                    "bridge_top_path_title_mentioned": float(audit["top_path_title_mentioned"]),
                }
            )
            oracle_action = max(
                ALL_ACTIONS,
                key=lambda action: (
                    actions[action]["f1"],
                    actions[action]["em"],
                    -ALL_ACTIONS.index(action),
                ),
            )
            rows.append(
                {
                    "id": row_id,
                    "dataset": dataset,
                    "question": str(base["question"]),
                    "features": features,
                    "actions": actions,
                    "direct_em": actions[DIRECT_ACTION]["em"],
                    "direct_f1": actions[DIRECT_ACTION]["f1"],
                    "oracle_em": actions[oracle_action]["em"],
                    "oracle_f1": actions[oracle_action]["f1"],
                    "oracle_action": oracle_action,
                }
            )
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)) or len(rows) < len(datasets) * 20:
        raise SystemExit("중복 ID가 있거나 정책 pilot 행이 너무 적습니다.")
    return rows


def _numeric(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows],
        dtype="float64",
    )


def _targets(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            [float(row["actions"][action]["f1"]) - float(row["direct_f1"]) for action in NON_DIRECT_ACTIONS]
            for row in rows
        ],
        dtype="float64",
    )


def _weights(rows: list[dict]) -> np.ndarray:
    counts = Counter(str(row["dataset"]) for row in rows)
    values = np.asarray([1.0 / counts[str(row["dataset"])] for row in rows], dtype="float64")
    return values / values.mean()


def _fit_models(rows: list[dict], args: argparse.Namespace, offset: int) -> BridgePolicyModels:
    numeric = _numeric(rows)
    questions = [str(row["question"]) for row in rows]
    targets = _targets(rows)
    labels = (targets.max(axis=1) > 1e-9).astype("int64")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("beneficial/non-beneficial 문항이 모두 필요합니다.")
    weights = _weights(rows)
    gate = QuestionEvidenceGate.fit(
        numeric,
        questions,
        labels,
        sample_weights=weights,
        seed=args.seed + offset,
        max_text_features=args.max_text_features,
    )
    value = ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed + offset + 1,
        n_jobs=-1,
    ).fit(numeric, targets, sample_weight=weights)
    return BridgePolicyModels(gate, value)


def _predict(models: BridgePolicyModels, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    numeric = _numeric(rows)
    questions = [str(row["question"]) for row in rows]
    return models.gate.predict_graph_probability(numeric, questions), np.asarray(models.value.predict(numeric))


def _select(
    rows: list[dict],
    probabilities: np.ndarray,
    advantages: np.ndarray,
    thresholds: Thresholds,
) -> tuple[list[str], list[float], list[float]]:
    actions: list[str] = []
    em_values: list[float] = []
    f1_values: list[float] = []
    for row, probability, estimates in zip(rows, probabilities, advantages, strict=True):
        index = int(np.argmax(estimates))
        expand = float(probability) > thresholds.gate and float(estimates[index]) > thresholds.advantage
        action = NON_DIRECT_ACTIONS[index] if expand else DIRECT_ACTION
        actions.append(action)
        em_values.append(float(row["actions"][action]["em"]))
        f1_values.append(float(row["actions"][action]["f1"]))
    return actions, em_values, f1_values


def _calibrate(rows, predictions, args, gate_grid, advantage_grid) -> Thresholds:
    probabilities, advantages = predictions
    return max(
        (Thresholds(float(gate), float(advantage)) for gate in gate_grid for advantage in advantage_grid),
        key=lambda thresholds: (
            _selection_objective(
                rows,
                _select(rows, probabilities, advantages, thresholds)[2],
                args.datasets,
                args.dataset_harm_penalty,
            ),
            thresholds.gate,
            thresholds.advantage,
        ),
    )


def _nested_audit(rows, args, gate_grid, advantage_grid):
    records: list[dict] = []
    gains: list[float] = []
    for fold in range(args.folds):
        test = [row for row in rows if _bucket(row["id"], "bridge-outer", args.folds) == fold]
        pool = [row for row in rows if _bucket(row["id"], "bridge-outer", args.folds) != fold]
        calibration = [
            row for row in pool if _fraction(row["id"], f"bridge-calibration:{fold}") < args.calibration_ratio
        ]
        calibration_ids = {row["id"] for row in calibration}
        fit = [row for row in pool if row["id"] not in calibration_ids]
        models = _fit_models(fit, args, fold * 10)
        thresholds = _calibrate(calibration, _predict(models, calibration), args, gate_grid, advantage_grid)
        probabilities, advantages = _predict(models, test)
        actions, em_values, f1_values = _select(test, probabilities, advantages, thresholds)
        fold_records = [
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "fold": fold,
                "selected_action": action,
                "gate_probability": float(probability),
                "predicted_advantage": float(np.max(estimate)),
                "direct_em": float(row["direct_em"]),
                "direct_f1": float(row["direct_f1"]),
                "policy_em": em,
                "policy_f1": f1,
                "oracle_em": float(row["oracle_em"]),
                "oracle_f1": float(row["oracle_f1"]),
            }
            for row, action, probability, estimate, em, f1 in zip(
                test, actions, probabilities, advantages, em_values, f1_values, strict=True
            )
        ]
        records.extend(fold_records)
        direct = _macro_value(fold_records, "direct_f1", args.datasets)
        policy = _macro_value(fold_records, "policy_f1", args.datasets)
        gains.append(policy - direct)
        _log(
            f"fold {fold + 1}/{args.folds}: fit/cal/test={len(fit)}/{len(calibration)}/{len(test)} "
            f"| Macro ΔF1={policy - direct:+.3f} | threshold="
            f"{thresholds.gate:.2f}/{thresholds.advantage:+.2f}"
        )
    if len(records) != len(rows) or len({record["id"] for record in records}) != len(rows):
        raise AssertionError("nested OOF가 모든 문항을 정확히 한 번 포함해야 합니다.")
    return records, gains


def _fit_final(rows, args, gate_grid, advantage_grid):
    probabilities = np.empty(len(rows), dtype="float64")
    advantages = np.empty((len(rows), len(NON_DIRECT_ACTIONS)), dtype="float64")
    for fold in range(args.folds):
        indices = [
            index for index, row in enumerate(rows)
            if _bucket(row["id"], "bridge-final", args.folds) == fold
        ]
        fit = [
            row for row in rows
            if _bucket(row["id"], "bridge-final", args.folds) != fold
        ]
        test = [rows[index] for index in indices]
        fold_probability, fold_advantage = _predict(_fit_models(fit, args, 10_000 + fold * 10), test)
        probabilities[indices] = fold_probability
        advantages[indices] = fold_advantage
    thresholds = _calibrate(rows, (probabilities, advantages), args, gate_grid, advantage_grid)
    return _fit_models(rows, args, 20_000), thresholds


def _report_static(rows: list[dict], datasets: Iterable[str]) -> None:
    _log("[고정 행동 Macro F1]")
    for action in ALL_ACTIONS:
        value = float(
            np.mean(
                [
                    np.mean([float(row["actions"][action]["f1"]) for row in rows if row["dataset"] == dataset])
                    for dataset in datasets
                ]
            )
        )
        _log(f"  {action}: {value:.3f}")


def _report_records(records: list[dict], datasets: Iterable[str]) -> dict:
    datasets = tuple(datasets)
    _log("[Nested OOF 실제 정책]")
    summary: dict[str, dict] = {}
    for name in (*datasets, "Macro"):
        selected = records if name == "Macro" else [row for row in records if row["dataset"] == name]
        if name == "Macro":
            direct_em = _macro_value(records, "direct_em", datasets)
            direct_f1 = _macro_value(records, "direct_f1", datasets)
            policy_em = _macro_value(records, "policy_em", datasets)
            policy_f1 = _macro_value(records, "policy_f1", datasets)
            oracle_f1 = _macro_value(records, "oracle_f1", datasets)
        else:
            direct_em = float(np.mean([row["direct_em"] for row in selected]))
            direct_f1 = float(np.mean([row["direct_f1"] for row in selected]))
            policy_em = float(np.mean([row["policy_em"] for row in selected]))
            policy_f1 = float(np.mean([row["policy_f1"] for row in selected]))
            oracle_f1 = float(np.mean([row["oracle_f1"] for row in selected]))
        expansion_rate = float(np.mean([row["selected_action"] != DIRECT_ACTION for row in selected]))
        _log(
            f"{name}: n={len(selected)} | Direct EM/F1={direct_em:.3f}/{direct_f1:.3f} "
            f"| Policy={policy_em:.3f}/{policy_f1:.3f} | gain="
            f"{policy_em - direct_em:+.3f}/{policy_f1 - direct_f1:+.3f} "
            f"| Oracle F1={oracle_f1:.3f} | expand={expansion_rate:.1%}"
        )
        summary[name] = {
            "n": len(selected),
            "direct_em": direct_em,
            "direct_f1": direct_f1,
            "policy_em": policy_em,
            "policy_f1": policy_f1,
            "em_gain": policy_em - direct_em,
            "f1_gain": policy_f1 - direct_f1,
            "oracle_f1": oracle_f1,
            "expansion_rate": expansion_rate,
        }
    _log(f"OOF 행동={dict(sorted(Counter(row['selected_action'] for row in records).items()))}")
    return summary


def _write_predictions(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for record in sorted(records, key=lambda item: (item["dataset"], item["id"])):
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
