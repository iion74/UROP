"""Train and nested-OOF evaluate a 4-hop Answer-F1 STOP/CONTINUE policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from experiments.train_answer_policy import _bucket, _fraction, _macro_value, _selection_objective
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from src.answer_policy import QuestionEvidenceGate


FORMAT_VERSION = 1
MAX_SEARCH_STEPS = 3
ACTIONS = ("vector", "planned_continue_1", "planned_continue_2", "planned_continue_3")
FEATURE_NAMES = (
    "completed_step_fraction",
    "reader_source_fraction",
    "memory_size_fraction",
    "previous_query_fraction",
    "last_added_score",
    "last_added_missing",
    "no_new_fraction",
    "query_unique_ratio",
    "last_query_repeated",
)


def main() -> None:
    args = _arguments()
    started = perf_counter()
    rows = _load_rows(Path(args.answer_root), args.datasets, args.split, args.max_rows_per_dataset)
    label_counts = Counter(
        label["label"]
        for row in rows
        for label in row["stop_labels"]
    )
    _log(
        f"4-hop STOP/CONTINUE 정책 학습 시작: {len(rows):,}문항 · 상태 {sum(label_counts.values()):,}개 "
        f"· label={dict(sorted(label_counts.items()))} · nested folds={args.folds}"
    )
    _report_static(rows, args.datasets)
    threshold_grid = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    records, fold_gains = _nested_oof(rows, args, threshold_grid)
    audit = _report_records(records, args.datasets)
    final_gate, final_threshold = _fit_final(rows, args, threshold_grid)

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
            "policy_type": "answer_f1_stop_continue_v1",
            "max_evidence_hops": 4,
            "max_search_steps": MAX_SEARCH_STEPS,
            "feature_names": FEATURE_NAMES,
            "threshold": final_threshold,
            "gate": final_gate,
        },
        temporary,
    )
    temporary.replace(model_path)
    _write_predictions(predictions_path, records)
    manifest = {
        "format_version": FORMAT_VERSION,
        "policy_type": "answer_f1_stop_continue_v1",
        "model": str(model_path),
        "nested_predictions": str(predictions_path),
        "datasets": list(args.datasets),
        "split": args.split,
        "rows": len(rows),
        "dataset_counts": dict(sorted(Counter(str(row["dataset"]) for row in rows).items())),
        "state_label_counts": dict(sorted(label_counts.items())),
        "feature_names": list(FEATURE_NAMES),
        "target": "future_answer_f1_gain",
        "seed": args.seed,
        "folds": args.folds,
        "threshold": final_threshold,
        "nested_audit": audit,
        "fold_macro_gains": fold_gains,
        "row_fingerprint": hashlib.sha256(
            "\n".join(sorted(str(row["id"]) for row in rows)).encode("utf-8")
        ).hexdigest(),
    }
    model_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"정책 저장: {model_path} · threshold={final_threshold:.3f}")
    _log(f"전체 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4-hop Answer-F1 STOP/CONTINUE nested OOF trainer")
    parser.add_argument("--answer-root", default="data/benchmarks/multihop/stop_answer_oracle_v1")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-steps", type=int, default=33)
    parser.add_argument("--max-text-features", type=int, default=20_000)
    parser.add_argument("--dataset-harm-penalty", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="data/models/stop_continue_policy_1000.joblib")
    parser.add_argument("--predictions-out")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_rows_per_dataset < 0 or args.folds < 3:
        parser.error("max rows는 0 이상, folds는 3 이상이어야 합니다.")
    if not 0.0 < args.calibration_ratio < 0.5:
        parser.error("calibration ratio는 0과 0.5 사이여야 합니다.")
    if args.threshold_steps < 2 or not 0.0 <= args.threshold_min < args.threshold_max <= 1.0:
        parser.error("threshold grid 설정을 확인하세요.")
    if args.max_text_features < 100 or args.dataset_harm_penalty < 0.0:
        parser.error("text feature와 harm penalty 설정을 확인하세요.")
    return args


def _load_rows(root: Path, datasets: tuple[str, ...], split: str, maximum: int) -> list[dict]:
    rows = []
    for dataset in datasets:
        path = root / f"{dataset}_{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"STOP Answer Oracle 결과가 없습니다: {path}")
        dataset_rows = list(_iter_jsonl(path))
        if maximum:
            dataset_rows = dataset_rows[:maximum]
        for row in dataset_rows:
            if tuple(row.get("actions", {})) != ACTIONS:
                raise SystemExit(f"{dataset}: 4-hop action 형식이 아닙니다: {row.get('id')}")
            if len(row.get("stop_labels", [])) != MAX_SEARCH_STEPS:
                raise SystemExit(f"{dataset}: STOP label이 3개가 아닙니다: {row.get('id')}")
            if len(row.get("planner_audit", [])) != MAX_SEARCH_STEPS:
                raise SystemExit(f"{dataset}: policy state가 3개가 아닙니다: {row.get('id')}")
        rows.extend(dataset_rows)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)) or len(rows) < max(20, len(datasets) * 10):
        raise SystemExit("중복 ID가 있거나 정책 학습 행이 너무 적습니다.")
    return rows


def _state_rows(rows: list[dict]) -> list[dict]:
    states = []
    for row in rows:
        queries = [str(item.get("query", "")) for item in row["planner_audit"]]
        for step in range(MAX_SEARCH_STEPS):
            audit = row["planner_audit"][step]
            state = audit["policy_state"]
            prior_queries = queries[:step]
            numeric = _numeric_state(state, prior_queries)
            text = str(row["question"])
            if prior_queries:
                text += "\nPrevious searches: " + " | ".join(prior_queries)
            label = row["stop_labels"][step]
            states.append(
                {
                    "question_id": str(row["id"]),
                    "dataset": str(row["dataset"]),
                    "step": step,
                    "numeric": numeric,
                    "text": text,
                    "label": int(label["label"] == "CONTINUE"),
                    "f1_gain": float(label["f1_gain"]),
                }
            )
    return states


def _numeric_state(state: dict, prior_queries: list[str]) -> tuple[float, ...]:
    normalised = [_normalise_query(query) for query in prior_queries if _normalise_query(query)]
    unique_ratio = len(set(normalised)) / len(normalised) if normalised else 1.0
    last_repeated = float(bool(normalised) and normalised[-1] in normalised[:-1])
    last_score = state.get("last_added_score")
    return (
        min(1.0, float(state.get("completed_steps", 0)) / MAX_SEARCH_STEPS),
        min(1.0, float(state.get("reader_source_count", 0)) / 5.0),
        min(1.0, float(state.get("memory_size", 0)) / 40.0),
        min(1.0, float(state.get("previous_query_count", 0)) / MAX_SEARCH_STEPS),
        float(last_score) if last_score is not None else 0.0,
        float(last_score is None),
        min(1.0, float(state.get("consecutive_no_new", 0)) / MAX_SEARCH_STEPS),
        unique_ratio,
        last_repeated,
    )


def _normalise_query(text: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", text.casefold()).strip()


def _fit_gate(rows: list[dict], args, offset: int) -> QuestionEvidenceGate:
    states = _state_rows(rows)
    labels = np.asarray([state["label"] for state in states], dtype="int64")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("STOP과 CONTINUE 상태가 모두 필요합니다.")
    counts = Counter(state["dataset"] for state in states)
    weights = np.asarray([1.0 / counts[state["dataset"]] for state in states], dtype="float64")
    weights /= weights.mean()
    return QuestionEvidenceGate.fit(
        np.asarray([state["numeric"] for state in states], dtype="float64"),
        [state["text"] for state in states],
        labels,
        sample_weights=weights,
        seed=args.seed + offset,
        max_text_features=args.max_text_features,
    )


def _state_probabilities(gate: QuestionEvidenceGate, rows: list[dict]) -> dict[str, list[float]]:
    states = _state_rows(rows)
    probabilities = gate.predict_graph_probability(
        np.asarray([state["numeric"] for state in states], dtype="float64"),
        [state["text"] for state in states],
    )
    by_id: dict[str, list[float]] = {str(row["id"]): [0.0] * MAX_SEARCH_STEPS for row in rows}
    for state, probability in zip(states, probabilities, strict=True):
        by_id[state["question_id"]][state["step"]] = float(probability)
    return by_id


def _simulate(rows: list[dict], probabilities: dict[str, list[float]], threshold: float) -> list[dict]:
    records = []
    for row in rows:
        values = probabilities[str(row["id"])]
        stop_step = MAX_SEARCH_STEPS
        for step, probability in enumerate(values):
            if probability < threshold:
                stop_step = step
                break
        action = ACTIONS[stop_step]
        result = row["actions"][action]
        records.append(
            {
                "id": str(row["id"]),
                "dataset": str(row["dataset"]),
                "selected_action": action,
                "stop_step": stop_step,
                "continue_probabilities": values,
                "direct_em": float(row["direct_em"]),
                "direct_f1": float(row["direct_f1"]),
                "policy_em": float(result["em"]),
                "policy_f1": float(result["f1"]),
                "oracle_em": float(row["oracle_em"]),
                "oracle_f1": float(row["oracle_f1"]),
            }
        )
    return records


def _calibrate(rows, probabilities, args, threshold_grid) -> float:
    return float(
        max(
            threshold_grid,
            key=lambda threshold: (
                _selection_objective(
                    rows,
                    [record["policy_f1"] for record in _simulate(rows, probabilities, float(threshold))],
                    args.datasets,
                    args.dataset_harm_penalty,
                ),
                float(threshold),
            ),
        )
    )


def _nested_oof(rows, args, threshold_grid):
    records = []
    fold_gains = []
    for fold in range(args.folds):
        test = [row for row in rows if _bucket(str(row["id"]), "stop-outer", args.folds) == fold]
        pool = [row for row in rows if _bucket(str(row["id"]), "stop-outer", args.folds) != fold]
        calibration = [
            row for row in pool
            if _fraction(str(row["id"]), f"stop-calibration:{fold}") < args.calibration_ratio
        ]
        calibration_ids = {str(row["id"]) for row in calibration}
        fit = [row for row in pool if str(row["id"]) not in calibration_ids]
        gate = _fit_gate(fit, args, fold * 10)
        threshold = _calibrate(
            calibration,
            _state_probabilities(gate, calibration),
            args,
            threshold_grid,
        )
        fold_records = _simulate(test, _state_probabilities(gate, test), threshold)
        for record in fold_records:
            record["fold"] = fold
            record["threshold"] = threshold
        records.extend(fold_records)
        direct = _macro_value(fold_records, "direct_f1", args.datasets)
        policy = _macro_value(fold_records, "policy_f1", args.datasets)
        fold_gains.append(policy - direct)
        _log(
            f"fold {fold + 1}/{args.folds}: fit/cal/test={len(fit)}/{len(calibration)}/{len(test)} "
            f"| Macro ΔF1={policy - direct:+.4f} | threshold={threshold:.3f}"
        )
    if len(records) != len(rows) or len({record["id"] for record in records}) != len(rows):
        raise AssertionError("nested OOF가 모든 질문을 정확히 한 번 포함해야 합니다.")
    return records, fold_gains


def _fit_final(rows, args, threshold_grid):
    probabilities: dict[str, list[float]] = {}
    for fold in range(args.folds):
        test = [row for row in rows if _bucket(str(row["id"]), "stop-final", args.folds) == fold]
        fit = [row for row in rows if _bucket(str(row["id"]), "stop-final", args.folds) != fold]
        probabilities.update(_state_probabilities(_fit_gate(fit, args, 10_000 + fold * 10), test))
    threshold = _calibrate(rows, probabilities, args, threshold_grid)
    return _fit_gate(rows, args, 20_000), threshold


def _report_static(rows, datasets: Iterable[str]) -> None:
    _log("[고정 STOP 시점 Macro F1]")
    for action in ACTIONS:
        value = float(
            np.mean(
                [
                    np.mean([float(row["actions"][action]["f1"]) for row in rows if row["dataset"] == dataset])
                    for dataset in datasets
                ]
            )
        )
        _log(f"  {action}: {value:.4f}")


def _report_records(records, datasets: Iterable[str]) -> dict:
    datasets = tuple(datasets)
    _log("[Nested OOF 실제 STOP 정책]")
    summary = {}
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
        continue_rate = float(np.mean([row["stop_step"] > 0 for row in selected]))
        _log(
            f"{name}: n={len(selected):,} | Direct EM/F1={direct_em:.4f}/{direct_f1:.4f} "
            f"| Policy={policy_em:.4f}/{policy_f1:.4f} | gain="
            f"{policy_em - direct_em:+.4f}/{policy_f1 - direct_f1:+.4f} "
            f"| Oracle F1={oracle_f1:.4f} | CONTINUE={continue_rate:.1%}"
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
            "continue_rate": continue_rate,
        }
    _log(f"OOF STOP step={dict(sorted(Counter(row['stop_step'] for row in records).items()))}")
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
