"""Calibrate and evaluate a seed-vote ensemble of STOP value policies.

Calibration uses only out-of-fold predictions from the train split.  Held-out
validation labels are loaded only after the vote thresholds and optional rule
weight have been frozen.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl
from experiments.train_stop_policy import ACTIONS, MAX_SEARCH_STEPS


MODEL_NAMES = (
    "stop_value_policy_v7_evidence_survival_bellman_em_f1_1000",
    "stop_value_policy_v7_evidence_survival_seed43_1000",
    "stop_value_policy_v7_evidence_survival_seed44_1000",
    "stop_value_policy_v7_evidence_survival_seed45_1000",
    "stop_value_policy_v7_evidence_survival_seed46_1000",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-OOF calibrated STOP value seed ensemble")
    parser.add_argument(
        "--train-answer-root",
        default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_train_1000",
    )
    parser.add_argument(
        "--validation-answer-root",
        default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_validation_2000",
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--max-train-rows-per-dataset", type=int, default=1000)
    parser.add_argument("--max-validation-rows-per-dataset", type=int, default=2000)
    parser.add_argument(
        "--mode",
        choices=("ensemble", "rule_guard"),
        default="ensemble",
        help="rule_guard adds the train-calibrated retrieval rule as a weighted vote.",
    )
    parser.add_argument("--search-cost", type=float, default=0.005)
    parser.add_argument("--f1-weight", type=float, default=0.5)
    parser.add_argument("--em-weight", type=float, default=0.5)
    parser.add_argument("--dataset-harm-penalty", type=float, default=2.0)
    parser.add_argument("--max-rule-weight", type=int, default=5)
    parser.add_argument(
        "--train-rule-predictions",
        default="data/benchmarks/multihop/rule_policy/retrieval_score_saturation_train_1000_oof_predictions.jsonl",
    )
    parser.add_argument(
        "--validation-rule-predictions",
        default="data/benchmarks/multihop/rule_policy/retrieval_score_saturation_validation_2000_predictions.jsonl",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-out", required=True)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    total = args.f1_weight + args.em_weight
    if args.max_train_rows_per_dataset < 1 or args.max_validation_rows_per_dataset < 1:
        parser.error("train/validation 행 수는 1 이상이어야 합니다.")
    if args.search_cost < 0 or args.dataset_harm_penalty < 0 or total <= 0:
        parser.error("비용/페널티는 0 이상이고 EM/F1 가중치 합은 0보다 커야 합니다.")
    if args.max_rule_weight < 1:
        parser.error("max rule weight는 1 이상이어야 합니다.")
    args.f1_weight /= total
    args.em_weight /= total
    return args


def _read_by_id(path: Path) -> dict[str, dict]:
    rows = {str(row["id"]): row for row in _iter_jsonl(path)}
    if not rows:
        raise SystemExit(f"예측 파일이 비어 있습니다: {path}")
    return rows


def _load_answers(root: Path, datasets: tuple[str, ...], split: str, maximum: int) -> list[dict]:
    rows = []
    for dataset in datasets:
        path = root / f"{dataset}_{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"Answer 결과가 없습니다: {path}")
        selected = list(_iter_jsonl(path))[:maximum]
        if len(selected) != maximum:
            raise SystemExit(f"{dataset}: 요청한 {maximum:,}개보다 Answer 행이 적습니다.")
        rows.extend(selected)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Answer 질문 ID가 중복됩니다.")
    return rows


def _prediction_paths() -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    manifests = tuple(Path("data/models") / f"{name}.manifest.json" for name in MODEL_NAMES)
    train = tuple(Path("data/models") / f"{name}.nested_predictions.jsonl" for name in MODEL_NAMES)
    validation_names = (
        "stop_value_v7_evidence_survival_seed42_validation_2000_predictions.jsonl",
        "stop_value_v7_evidence_survival_seed43_validation_2000_predictions.jsonl",
        "stop_value_v7_evidence_survival_seed44_validation_2000_predictions.jsonl",
        "stop_value_v7_evidence_survival_seed45_validation_2000_predictions.jsonl",
        "stop_value_v7_evidence_survival_seed46_validation_2000_predictions.jsonl",
    )
    validation = tuple(Path("data/benchmarks/multihop/policy_eval") / name for name in validation_names)
    return manifests, train, validation


def _load_prediction_sets(paths: tuple[Path, ...], expected_ids: set[str]) -> tuple[dict[str, dict], ...]:
    result = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"정책 예측이 없습니다: {path}")
        predictions = _read_by_id(path)
        if predictions.keys() != expected_ids:
            missing = next(iter(expected_ids - predictions.keys()), None)
            extra = next(iter(predictions.keys() - expected_ids), None)
            raise SystemExit(f"정책 예측 ID 불일치: {path}, missing={missing}, extra={extra}")
        result.append(predictions)
    return tuple(result)


def _vote_matrix(
    rows: list[dict],
    predictions: tuple[dict[str, dict], ...],
    final_thresholds: tuple[tuple[float, ...], ...] | None = None,
) -> np.ndarray:
    votes = np.zeros((len(rows), MAX_SEARCH_STEPS), dtype="int64")
    for row_index, row in enumerate(rows):
        row_id = str(row["id"])
        for model_index, model_predictions in enumerate(predictions):
            prediction = model_predictions[row_id]
            thresholds = final_thresholds[model_index] if final_thresholds else prediction["threshold"]
            advantages = prediction["predicted_advantages"]
            if len(thresholds) != MAX_SEARCH_STEPS or len(advantages) != MAX_SEARCH_STEPS:
                raise SystemExit(f"단계 수가 맞지 않습니다: {row_id}")
            votes[row_index] += np.asarray(
                [float(value) > float(threshold) for value, threshold in zip(advantages, thresholds, strict=True)],
                dtype="int64",
            )
    return votes


def _rule_matrix(rows: list[dict], path: Path) -> np.ndarray:
    predictions = _read_by_id(path)
    expected = {str(row["id"]) for row in rows}
    if predictions.keys() != expected:
        raise SystemExit(f"규칙 예측 ID 불일치: {path}")
    return np.asarray(
        [
            [int(predictions[str(row["id"])]["stop_step"]) > step for step in range(MAX_SEARCH_STEPS)]
            for row in rows
        ],
        dtype="int64",
    )


def _simulate(score: np.ndarray, thresholds: tuple[int, ...]) -> np.ndarray:
    if score.ndim != 2 or score.shape[1] != MAX_SEARCH_STEPS:
        raise ValueError("score matrix의 단계 수가 맞지 않습니다.")
    stop_steps = np.full(len(score), MAX_SEARCH_STEPS, dtype="int64")
    active = np.ones(len(score), dtype="bool")
    for step in range(MAX_SEARCH_STEPS):
        continue_search = score[:, step] >= thresholds[step]
        stopping = active & ~continue_search
        stop_steps[stopping] = step
        active &= continue_search
    return stop_steps


def _selected(rows: list[dict], stop_steps: np.ndarray, metric: str) -> np.ndarray:
    return np.asarray(
        [float(row["actions"][ACTIONS[int(step)]][metric]) for row, step in zip(rows, stop_steps, strict=True)],
        dtype="float64",
    )


def _objective(rows: list[dict], stop_steps: np.ndarray, args) -> tuple[float, float, float]:
    em = _selected(rows, stop_steps, "em")
    f1 = _selected(rows, stop_steps, "f1")
    selected_utility = args.em_weight * em + args.f1_weight * f1 - args.search_cost * stop_steps
    direct_utility = np.asarray(
        [args.em_weight * float(row["direct_em"]) + args.f1_weight * float(row["direct_f1"]) for row in rows]
    )
    dataset_values = []
    harms = []
    for dataset in args.datasets:
        mask = np.asarray([str(row["dataset"]) == dataset for row in rows], dtype="bool")
        policy_value = float(selected_utility[mask].mean())
        direct_value = float(direct_utility[mask].mean())
        dataset_values.append(policy_value)
        harms.append(max(0.0, direct_value - policy_value))
    protected = float(np.mean(dataset_values) - args.dataset_harm_penalty * sum(harms))
    return protected, float(np.mean(f1)), float(np.mean(stop_steps))


def _calibrate(rows: list[dict], votes: np.ndarray, rule: np.ndarray, args) -> tuple[int, tuple[int, ...], dict]:
    weights = (0,) if args.mode == "ensemble" else tuple(range(1, args.max_rule_weight + 1))
    best = None
    candidates = 0
    for rule_weight in weights:
        score = votes + rule_weight * rule
        maximum = len(MODEL_NAMES) + rule_weight
        for thresholds in itertools.product(range(maximum + 2), repeat=MAX_SEARCH_STEPS):
            stop_steps = _simulate(score, thresholds)
            protected, f1, average_steps = _objective(rows, stop_steps, args)
            # Freeze one deterministic preference order: protected utility,
            # answer F1, fewer searches, then stricter/lower-complexity rule.
            key = (protected, f1, -average_steps, thresholds, -rule_weight)
            candidates += 1
            if best is None or key > best[0]:
                best = (key, rule_weight, thresholds, stop_steps)
    assert best is not None
    key, rule_weight, thresholds, stop_steps = best
    audit = {
        "candidates": candidates,
        "protected_utility": key[0],
        "answer_f1": key[1],
        "average_extra_searches": -key[2],
        "metrics": _summary(rows, stop_steps, args.datasets),
    }
    return rule_weight, tuple(int(value) for value in thresholds), audit


def _records(
    rows: list[dict],
    votes: np.ndarray,
    rule: np.ndarray,
    rule_weight: int,
    thresholds: tuple[int, ...],
) -> list[dict]:
    stop_steps = _simulate(votes + rule_weight * rule, thresholds)
    records = []
    for index, (row, step) in enumerate(zip(rows, stop_steps, strict=True)):
        action = ACTIONS[int(step)]
        selected = row["actions"][action]
        records.append(
            {
                "id": str(row["id"]),
                "dataset": str(row["dataset"]),
                "selected_action": action,
                "stop_step": int(step),
                "model_continue_votes": votes[index].tolist(),
                "rule_continue": rule[index].tolist(),
                "rule_weight": rule_weight,
                "vote_thresholds": list(thresholds),
                "direct_em": float(row["direct_em"]),
                "direct_f1": float(row["direct_f1"]),
                "policy_em": float(selected["em"]),
                "policy_f1": float(selected["f1"]),
                "source_recall": float(selected["source_recall"]),
                "all_support_recall": float(selected["all_support_recall"]),
                "context_characters": int(selected.get("context_characters", 0)),
                "reader_latency_seconds": float(selected.get("reader_latency_seconds", 0.0)),
                "oracle_f1": float(row["oracle_f1"]),
            }
        )
    return records


def _summary(rows: list[dict], stop_steps: np.ndarray, datasets: tuple[str, ...]) -> dict:
    summary = {}
    for dataset in datasets:
        indices = [index for index, row in enumerate(rows) if str(row["dataset"]) == dataset]
        selected_steps = stop_steps[indices]
        selected_rows = [rows[index] for index in indices]
        em = _selected(selected_rows, selected_steps, "em")
        f1 = _selected(selected_rows, selected_steps, "f1")
        summary[dataset] = {
            "n": len(indices),
            "em": float(em.mean()),
            "f1": float(f1.mean()),
            "average_extra_searches": float(selected_steps.mean()),
            "stop_steps": {
                str(step): int(np.sum(selected_steps == step)) for step in range(MAX_SEARCH_STEPS + 1)
            },
        }
    return summary


def _paired_bootstrap(records: list[dict], datasets: tuple[str, ...], samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    for dataset in datasets:
        selected = [row for row in records if row["dataset"] == dataset]
        dataset_result = {}
        for metric in ("em", "f1"):
            differences = np.asarray(
                [float(row[f"policy_{metric}"]) - float(row[f"direct_{metric}"]) for row in selected]
            )
            draws = np.empty(samples, dtype="float64")
            for start in range(0, samples, 256):
                end = min(samples, start + 256)
                indices = rng.integers(0, len(differences), size=(end - start, len(differences)))
                draws[start:end] = differences[indices].mean(axis=1)
            dataset_result[metric] = {
                "observed_gain": float(differences.mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "probability_positive": float(np.mean(draws > 0.0)),
            }
        result[dataset] = dataset_result
    return result


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in sorted(records, key=lambda item: (item["dataset"], item["id"])):
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    manifests, train_paths, validation_paths = _prediction_paths()
    train_rows = _load_answers(Path(args.train_answer_root), args.datasets, "train", args.max_train_rows_per_dataset)
    validation_rows = _load_answers(
        Path(args.validation_answer_root), args.datasets, "validation", args.max_validation_rows_per_dataset
    )
    train_ids = {str(row["id"]) for row in train_rows}
    validation_ids = {str(row["id"]) for row in validation_rows}
    train_predictions = _load_prediction_sets(train_paths, train_ids)
    validation_predictions = _load_prediction_sets(validation_paths, validation_ids)
    final_thresholds = tuple(
        tuple(float(value) for value in json.loads(path.read_text(encoding="utf-8"))["threshold"])
        for path in manifests
    )
    train_votes = _vote_matrix(train_rows, train_predictions)
    validation_votes = _vote_matrix(validation_rows, validation_predictions, final_thresholds)
    train_rule = _rule_matrix(train_rows, Path(args.train_rule_predictions))
    validation_rule = _rule_matrix(validation_rows, Path(args.validation_rule_predictions))
    rule_weight, thresholds, train_audit = _calibrate(train_rows, train_votes, train_rule, args)
    records = _records(validation_rows, validation_votes, validation_rule, rule_weight, thresholds)
    stop_steps = np.asarray([row["stop_step"] for row in records], dtype="int64")
    result = {
        "format_version": 1,
        "policy": "five_seed_stop_value_vote_ensemble",
        "mode": args.mode,
        "models": list(MODEL_NAMES),
        "calibration_split": "train_out_of_fold_predictions",
        "validation_labels_used_for_calibration": False,
        "rule_weight": rule_weight,
        "vote_thresholds": list(thresholds),
        "utility": {
            "f1_weight": args.f1_weight,
            "em_weight": args.em_weight,
            "search_cost": args.search_cost,
            "dataset_harm_penalty": args.dataset_harm_penalty,
        },
        "train_oof_audit": train_audit,
        "validation": _summary(validation_rows, stop_steps, args.datasets),
        "paired_bootstrap_vs_direct": _paired_bootstrap(
            records, args.datasets, args.bootstrap_samples, args.bootstrap_seed
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    _write_jsonl(Path(args.predictions_out), records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
