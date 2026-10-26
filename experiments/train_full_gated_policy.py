"""Train the final one-shot retrieval policy from all benchmark train rows.

The expensive frozen-reader action labels are produced by
``evaluate_context_answer_oracle``.  This command consumes every available
row, performs leakage-free nested cross-validation, calibrates the graph gate
from out-of-fold predictions, and finally fits one deployable policy.

Dataset mixing follows the common temperature form.  If dataset ``d`` has
``n_d`` rows, each row receives weight ``n_d ** (alpha - 1)``:

* alpha=0.0 (default): equal total contribution per dataset, no rows removed
* alpha=0.5: square-root/temperature compromise
* alpha=1.0: natural example-proportional mixing
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

from experiments.evaluate_answer_oracle import DIRECT_ACTION, OUTPUT_FORMAT_VERSION
from experiments.evaluate_context_answer_oracle import CONTEXT_MODE
from experiments.train_answer_policy import (
    ONE_SHOT_GRAPH_ACTIONS,
    _bucket,
    _fraction,
    _initial_features,
    _macro_value,
    _one_shot_targets,
    _selection_objective,
)
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from src.answer_policy import QuestionEvidenceGate
from src.policy import POLICY_FEATURE_NAMES


MODEL_FORMAT_VERSION = 2
POLICY_TYPE = "question_evidence_gated_answer_policy"


@dataclass(frozen=True)
class OneShotModels:
    gate: QuestionEvidenceGate
    value: ExtraTreesRegressor


def main() -> None:
    args = _arguments()
    started = perf_counter()
    rows, source_manifests = _load_rows(
        Path(args.rows_root),
        args.datasets,
        args.split,
        allow_partial=args.allow_partial_rows,
    )
    counts = Counter(str(row["dataset"]) for row in rows)
    _log(
        f"전체 one-shot Gated 정책 학습 시작: {len(rows):,}문항 · "
        f"raw counts={dict(counts)} · mixing alpha={args.mixing_alpha:.2f} · "
        f"nested folds={args.folds}"
    )
    totals = _effective_dataset_totals(rows, args.mixing_alpha)
    _log(
        "효과적 데이터셋 가중치 총합: "
        + ", ".join(f"{dataset}={totals[dataset]:.1f}" for dataset in args.datasets)
    )
    _log("QA reader·retriever·Graph executor는 동결하고, 행동 선택 gate/value model만 학습합니다.")

    grid = np.linspace(args.gate_threshold_min, args.gate_threshold_max, args.gate_threshold_steps)
    records, fold_summaries = _nested_audit(rows, args, grid)
    audit = _report_records(records, args.datasets, heading="중첩 교차검증 OOF")

    models, threshold = _fit_final(rows, args, grid)
    frozen_static_action = _best_static_action(rows, args.datasets)
    answer_oracle_signature = _retrieval_signature(source_manifests[0])
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
            "format_version": MODEL_FORMAT_VERSION,
            "policy_type": POLICY_TYPE,
            "training_mode": "full_data_one_shot_nested_oof_v1",
            "feature_names": POLICY_FEATURE_NAMES,
            "one_shot_actions": ONE_SHOT_GRAPH_ACTIONS,
            "thresholds": {"one_shot": threshold},
            "initial_gate": models.gate,
            "one_shot_value_model": models.value,
            "frozen_static_action": frozen_static_action,
            "answer_oracle_signature": answer_oracle_signature,
        },
        temporary,
    )
    temporary.replace(model_path)
    _write_predictions(predictions_path, records)
    _write_manifest(
        model_path,
        predictions_path,
        rows,
        source_manifests,
        args,
        threshold=threshold,
        frozen_static_action=frozen_static_action,
        audit=audit,
        fold_summaries=fold_summaries,
    )
    _log(f"\n최종 정책 저장: {model_path}")
    _log(f"OOF 예측 저장: {predictions_path}")
    _log(f"동결 gate threshold={threshold:.3f} · train에서 선택한 static baseline={frozen_static_action}")
    _log(f"전체 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="전체 3-dataset Answer-aware one-shot 검색 정책 학습")
    parser.add_argument(
        "--rows-root",
        default="data/benchmarks/multihop/context_answer_oracle",
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument(
        "--mixing-alpha",
        type=float,
        default=0.0,
        help="0=dataset-balanced, 0.5=temperature compromise, 1=example-proportional",
    )
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
    parser.add_argument(
        "--model-out",
        default="data/models/full_gated_one_shot_seed42.joblib",
    )
    parser.add_argument("--predictions-out")
    parser.add_argument(
        "--allow-partial-rows",
        action="store_true",
        help="smoke/pilot에서만 사용; 완전한 train manifest 검사를 해제",
    )
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if not args.allow_partial_rows and args.datasets != SUPPORTED_DATASETS:
        parser.error("최종 학습은 hotpotqa musique 2wiki 전체를 그 순서대로 사용해야 합니다.")
    if not 0.0 <= args.mixing_alpha <= 1.0:
        parser.error("mixing alpha는 0과 1 사이여야 합니다.")
    if args.folds < 3 or not 0.0 < args.calibration_ratio < 0.5:
        parser.error("folds는 3 이상, calibration ratio는 0과 0.5 사이여야 합니다.")
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


def _load_rows(
    root: Path,
    datasets: tuple[str, ...],
    split: str,
    *,
    allow_partial: bool,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    manifests: list[dict] = []
    for dataset in datasets:
        path = root / f"{dataset}_{split}.jsonl"
        manifest_path = path.with_suffix(".manifest.json")
        if not path.exists() or not manifest_path.exists():
            raise SystemExit(f"Answer Oracle 결과가 없습니다: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(manifest, dataset, split, manifest_path, allow_partial=allow_partial)
        dataset_rows: list[dict] = []
        for raw in _iter_jsonl(path):
            _validate_row(raw, dataset, path)
            dataset_rows.append(_slim_row(raw))
        if not dataset_rows:
            raise SystemExit(f"학습할 Oracle 행이 없습니다: {path}")
        if len(dataset_rows) != int(manifest.get("completed_rows", -1)):
            raise SystemExit(f"JSONL/manifest 행 수가 다릅니다: {path}")
        rows.extend(dataset_rows)
        manifests.append(manifest)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("학습 Oracle 행에 중복 ID가 있습니다.")
    signatures = [_retrieval_signature(manifest) for manifest in manifests]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise SystemExit("세 데이터셋의 reader/검색 설정이 서로 다릅니다.")
    if len(rows) < len(datasets) * 10:
        raise SystemExit("중첩 교차검증에 필요한 학습 행이 너무 적습니다.")
    return rows, manifests


def _validate_manifest(
    manifest: Mapping,
    dataset: str,
    split: str,
    path: Path,
    *,
    allow_partial: bool,
) -> None:
    required_actions = {DIRECT_ACTION, *ONE_SHOT_GRAPH_ACTIONS}
    valid = (
        int(manifest.get("format_version", -1)) == OUTPUT_FORMAT_VERSION
        and manifest.get("dataset") == dataset
        and manifest.get("split") == split
        and manifest.get("status") == "ready"
        and manifest.get("context_mode") == CONTEXT_MODE
        and required_actions.issubset(manifest.get("actions", ()))
        and int(manifest.get("completed_rows", -1)) == int(manifest.get("requested_rows", -2))
    )
    if not allow_partial:
        valid = valid and int(manifest.get("max_queries_per_dataset", -1)) == 0
    if not valid:
        raise SystemExit(f"완료된 공식-context train Oracle manifest가 아닙니다: {path}")


def _validate_row(row: Mapping, dataset: str, path: Path) -> None:
    required_actions = {DIRECT_ACTION, *ONE_SHOT_GRAPH_ACTIONS}
    if (
        int(row.get("format_version", -1)) != OUTPUT_FORMAT_VERSION
        or row.get("dataset") != dataset
        or not required_actions.issubset(row.get("actions", {}))
        or any(name not in row.get("features", {}) for name in POLICY_FEATURE_NAMES)
    ):
        raise SystemExit(f"지원하지 않는 Oracle 행입니다: {path} · {row.get('id')}")


def _slim_row(row: Mapping) -> dict:
    return {
        "id": str(row["id"]),
        "dataset": str(row["dataset"]),
        "question": str(row["question"]),
        "features": {name: float(row["features"][name]) for name in POLICY_FEATURE_NAMES},
        "actions": {
            action: {
                "em": float(row["actions"][action]["em"]),
                "f1": float(row["actions"][action]["f1"]),
            }
            for action in (DIRECT_ACTION, *ONE_SHOT_GRAPH_ACTIONS)
        },
        "direct_em": float(row["direct_em"]),
        "direct_f1": float(row["direct_f1"]),
        "oracle_em": float(row["oracle_em"]),
        "oracle_f1": float(row["oracle_f1"]),
    }


def _retrieval_signature(manifest: Mapping) -> dict:
    reader = manifest.get("reader", {})
    return {
        key: manifest.get(key)
        for key in (
            "context_mode",
            "top_k",
            "initial_candidates",
            "context_character_budget",
            "corpus_seed",
            "query_seed",
            "query_sampling_version",
            "actions",
            "chunk_size",
            "chunk_overlap",
            "embedding_model",
            "reranker_model",
            "reranker_candidates",
            "graph_format_version",
            "graph_semantic_neighbors",
            "graph_title_bridges",
            "graph_max_title_bridges_per_chunk",
            "router_seed_count",
            "graph_small_budget",
            "graph_large_budget",
            "graph_expansion_decay",
            "graph_rerank_prior_weight",
            "graph_source_diversity_bonus",
            "graph_min_expanded_results",
        )
    } | {
        "reader": {
            key: reader.get(key)
            for key in (
                "model",
                "model_digest",
                "max_tokens",
                "seed",
                "temperature",
                "prompt_version",
            )
        }
    }


def _sample_weights(rows: list[dict], alpha: float) -> np.ndarray:
    counts = Counter(str(row["dataset"]) for row in rows)
    raw = np.asarray(
        [counts[str(row["dataset"])] ** (alpha - 1.0) for row in rows],
        dtype="float64",
    )
    return raw / raw.mean()


def _effective_dataset_totals(rows: list[dict], alpha: float) -> dict[str, float]:
    weights = _sample_weights(rows, alpha)
    return {
        dataset: float(sum(weight for row, weight in zip(rows, weights, strict=True) if row["dataset"] == dataset))
        for dataset in dict.fromkeys(str(row["dataset"]) for row in rows)
    }


def _value_model(args: argparse.Namespace, seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=seed,
        n_jobs=-1,
    )


def _fit_models(rows: list[dict], args: argparse.Namespace, seed_offset: int) -> OneShotModels:
    weights = _sample_weights(rows, args.mixing_alpha)
    numeric = _initial_features(rows)
    questions = [str(row["question"]) for row in rows]
    labels = np.asarray(
        [
            max(float(row["actions"][action]["f1"]) for action in ONE_SHOT_GRAPH_ACTIONS)
            > float(row["direct_f1"]) + 1e-9
            for row in rows
        ],
        dtype="int64",
    )
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Graph-beneficial/non-beneficial 학습 문항이 모두 필요합니다.")
    gate = QuestionEvidenceGate.fit(
        numeric,
        questions,
        labels,
        sample_weights=weights,
        seed=args.seed + seed_offset,
        max_text_features=args.max_text_features,
    )
    value = _value_model(args, args.seed + seed_offset + 1).fit(
        numeric,
        _one_shot_targets(rows),
        sample_weight=weights,
    )
    return OneShotModels(gate, value)


def _predict(models: OneShotModels, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    numeric = _initial_features(rows)
    questions = [str(row["question"]) for row in rows]
    return (
        models.gate.predict_graph_probability(numeric, questions),
        np.asarray(models.value.predict(numeric)),
    )


def _selected(
    rows: list[dict],
    probabilities: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> tuple[list[str], list[float], list[float]]:
    actions: list[str] = []
    em_values: list[float] = []
    f1_values: list[float] = []
    for row, probability, estimates in zip(rows, probabilities, values, strict=True):
        index = int(np.argmax(estimates))
        action = ONE_SHOT_GRAPH_ACTIONS[index] if float(probability) > threshold else DIRECT_ACTION
        actions.append(action)
        em_values.append(float(row["actions"][action]["em"]))
        f1_values.append(float(row["actions"][action]["f1"]))
    return actions, em_values, f1_values


def _calibrate(
    rows: list[dict],
    predictions: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
    grid: np.ndarray,
) -> float:
    probabilities, values = predictions
    return float(
        max(
            grid,
            key=lambda threshold: (
                _selection_objective(
                    rows,
                    _selected(rows, probabilities, values, float(threshold))[2],
                    args.datasets,
                    args.dataset_harm_penalty,
                ),
                float(threshold),
            ),
        )
    )


def _nested_audit(
    rows: list[dict],
    args: argparse.Namespace,
    grid: np.ndarray,
) -> tuple[list[dict], list[float]]:
    records: list[dict] = []
    gains: list[float] = []
    for fold in range(args.folds):
        test = [row for row in rows if _bucket(row["id"], "outer", args.folds) == fold]
        pool = [row for row in rows if _bucket(row["id"], "outer", args.folds) != fold]
        calibration = [
            row for row in pool if _fraction(row["id"], f"calibration:{fold}") < args.calibration_ratio
        ]
        calibration_ids = {row["id"] for row in calibration}
        fit = [row for row in pool if row["id"] not in calibration_ids]
        if not fit or not calibration or not test:
            raise ValueError(f"fold {fold}의 fit/calibration/test 중 하나가 비었습니다.")
        models = _fit_models(fit, args, seed_offset=fold * 10)
        threshold = _calibrate(calibration, _predict(models, calibration), args, grid)
        fold_records = _records(test, _predict(models, test), threshold, fold)
        records.extend(fold_records)
        direct = _macro_value(fold_records, "direct_f1", args.datasets)
        policy = _macro_value(fold_records, "policy_f1", args.datasets)
        gains.append(policy - direct)
        _log(
            f"fold {fold + 1}/{args.folds}: fit/cal/test="
            f"{len(fit):,}/{len(calibration):,}/{len(test):,} | Macro ΔF1={policy - direct:+.4f} "
            f"| gate={threshold:.3f}"
        )
    if len(records) != len(rows) or len({record["id"] for record in records}) != len(rows):
        raise AssertionError("nested OOF가 모든 문항을 정확히 한 번씩 포함하지 않습니다.")
    return records, gains


def _fit_final(
    rows: list[dict],
    args: argparse.Namespace,
    grid: np.ndarray,
) -> tuple[OneShotModels, float]:
    probabilities = np.empty(len(rows), dtype="float64")
    values = np.empty((len(rows), len(ONE_SHOT_GRAPH_ACTIONS)), dtype="float64")
    for fold in range(args.folds):
        indices = [
            index for index, row in enumerate(rows)
            if _bucket(row["id"], "final-oof", args.folds) == fold
        ]
        fit = [
            row for row in rows
            if _bucket(row["id"], "final-oof", args.folds) != fold
        ]
        test = [rows[index] for index in indices]
        predicted_probability, predicted_value = _predict(
            _fit_models(fit, args, seed_offset=10_000 + fold * 10),
            test,
        )
        probabilities[indices] = predicted_probability
        values[indices] = predicted_value
        _log(f"최종 threshold OOF {fold + 1}/{args.folds} 완료: fit/test={len(fit):,}/{len(test):,}")
    threshold = _calibrate(rows, (probabilities, values), args, grid)
    return _fit_models(rows, args, seed_offset=20_000), threshold


def _records(
    rows: list[dict],
    predictions: tuple[np.ndarray, np.ndarray],
    threshold: float,
    fold: int,
) -> list[dict]:
    actions, em_values, f1_values = _selected(rows, *predictions, threshold)
    return [
        {
            "id": row["id"],
            "dataset": row["dataset"],
            "fold": fold,
            "gate_threshold": threshold,
            "selected_action": action,
            "direct_em": float(row["direct_em"]),
            "direct_f1": float(row["direct_f1"]),
            "policy_em": em,
            "policy_f1": f1,
            "oracle_em": float(row["oracle_em"]),
            "oracle_f1": float(row["oracle_f1"]),
        }
        for row, action, em, f1 in zip(rows, actions, em_values, f1_values, strict=True)
    ]


def _best_static_action(rows: list[dict], datasets: Iterable[str]) -> str:
    datasets = tuple(datasets)
    return max(
        ONE_SHOT_GRAPH_ACTIONS,
        key=lambda action: (
            np.mean(
                [
                    np.mean([float(row["actions"][action]["f1"]) for row in rows if row["dataset"] == dataset])
                    for dataset in datasets
                ]
            ),
            -ONE_SHOT_GRAPH_ACTIONS.index(action),
        ),
    )


def _report_records(records: list[dict], datasets: Iterable[str], *, heading: str) -> dict:
    datasets = tuple(datasets)
    _log(f"\n[{heading}]")
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
        graph_rate = float(np.mean([row["selected_action"] != DIRECT_ACTION for row in selected]))
        _log(
            f"{name}: n={len(selected):,} | Direct EM/F1={direct_em:.3f}/{direct_f1:.3f} "
            f"| Policy EM/F1={policy_em:.3f}/{policy_f1:.3f} "
            f"| gain={policy_em - direct_em:+.3f}/{policy_f1 - direct_f1:+.3f} "
            f"| Oracle F1={oracle_f1:.3f} | Graph={graph_rate:.1%}"
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
            "graph_rate": graph_rate,
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


def _write_manifest(
    model_path: Path,
    predictions_path: Path,
    rows: list[dict],
    source_manifests: list[dict],
    args: argparse.Namespace,
    *,
    threshold: float,
    frozen_static_action: str,
    audit: dict,
    fold_summaries: list[float],
) -> None:
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "policy_type": POLICY_TYPE,
        "training_mode": "full_data_one_shot_nested_oof_v1",
        "model": str(model_path),
        "nested_predictions": str(predictions_path),
        "datasets": list(args.datasets),
        "split": args.split,
        "rows": len(rows),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "all_available_rows_required": not args.allow_partial_rows,
        "mixing": {
            "formula": "per_row_weight = n_dataset ** (alpha - 1), normalized_to_mean_1",
            "alpha": args.mixing_alpha,
            "effective_dataset_weight_totals": _effective_dataset_totals(rows, args.mixing_alpha),
        },
        "target_metric": "answer_f1",
        "seed": args.seed,
        "folds": args.folds,
        "calibration_ratio": args.calibration_ratio,
        "final_threshold_calibration": f"{args.folds}-fold-oof",
        "trees": args.trees,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "max_text_features": args.max_text_features,
        "dataset_harm_penalty": args.dataset_harm_penalty,
        "thresholds": {"one_shot": threshold},
        "frozen_static_action": frozen_static_action,
        "answer_oracle_signature": _retrieval_signature(source_manifests[0]),
        "nested_audit": audit,
        "fold_macro_f1_gains": fold_summaries,
        "source_oracle_manifests": [
            {
                "dataset": manifest["dataset"],
                "processed_fingerprint": manifest["processed_fingerprint"],
                "completed_rows": manifest["completed_rows"],
                "context_mode": manifest["context_mode"],
            }
            for manifest in source_manifests
        ],
        "row_fingerprint": hashlib.sha256(
            "\n".join(sorted(str(row["id"]) for row in rows)).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path = model_path.with_suffix(".manifest.json")
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
