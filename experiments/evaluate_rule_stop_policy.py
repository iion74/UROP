"""Calibrate and evaluate a transparent retrieval-score STOP rule.

The baseline deliberately contains no learned model and never exposes gold
answers, supporting facts, or hop counts at inference time.  It observes only
retrieval scores and search-state saturation signals:

* At the initial state, continue when the second-best retrieved source is not
  confident enough.
* After an expansion, continue only while the newly added source score remains
  above a step-specific floor.
* Stop immediately when the planner reports sufficient evidence, no new
  evidence, or a repeated rewrite.

All three score thresholds are calibrated on the train split and then frozen
for held-out validation.  Question-level out-of-fold train results are emitted
as an overfitting audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.train_answer_policy import _bucket
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from experiments.train_stop_policy import ACTIONS, MAX_SEARCH_STEPS


FORMAT_VERSION = 1
RULE_NAME = "retrieval_score_saturation_v1"


@dataclass(frozen=True)
class RuleBatch:
    rows: tuple[dict, ...]
    scores: np.ndarray
    hard_stop: np.ndarray
    action_em: np.ndarray
    action_f1: np.ndarray
    action_source_recall: np.ndarray
    action_all_support_recall: np.ndarray
    datasets: np.ndarray


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-calibrated retrieval-score STOP rule baseline")
    parser.add_argument(
        "--train-answer-root",
        default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_train_1000",
    )
    parser.add_argument(
        "--train-trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_train_1000",
    )
    parser.add_argument(
        "--validation-answer-root",
        default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_validation_2000",
    )
    parser.add_argument(
        "--validation-trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--max-train-rows-per-dataset", type=int, default=1000)
    parser.add_argument("--max-validation-rows-per-dataset", type=int, default=2000)
    parser.add_argument("--threshold-steps", type=int, default=51)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--f1-weight", type=float, default=0.5)
    parser.add_argument("--em-weight", type=float, default=0.5)
    parser.add_argument("--search-cost", type=float, default=0.002)
    parser.add_argument("--dataset-harm-penalty", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/benchmarks/multihop/rule_policy/retrieval_score_saturation_validation_2000.json",
    )
    parser.add_argument(
        "--predictions-out",
        default="data/benchmarks/multihop/rule_policy/retrieval_score_saturation_validation_2000_predictions.jsonl",
    )
    parser.add_argument(
        "--train-oof-predictions-out",
        default="data/benchmarks/multihop/rule_policy/retrieval_score_saturation_train_1000_oof_predictions.jsonl",
    )
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_train_rows_per_dataset < 1 or args.max_validation_rows_per_dataset < 1:
        parser.error("train/validation 최대 행 수는 1 이상이어야 합니다.")
    if args.threshold_steps < 3 or args.folds < 3:
        parser.error("threshold steps와 folds는 3 이상이어야 합니다.")
    if args.search_cost < 0.0 or args.dataset_harm_penalty < 0.0:
        parser.error("검색 비용과 데이터셋 손실 페널티는 0 이상이어야 합니다.")
    if args.f1_weight < 0.0 or args.em_weight < 0.0 or args.f1_weight + args.em_weight <= 0.0:
        parser.error("EM/F1 가중치는 0 이상이고 합은 0보다 커야 합니다.")
    total = args.f1_weight + args.em_weight
    args.f1_weight /= total
    args.em_weight /= total
    return args


def _load_rows(
    answer_root: Path,
    trajectory_root: Path,
    datasets: tuple[str, ...],
    split: str,
    maximum: int,
) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    trajectories: dict[str, dict] = {}
    for dataset in datasets:
        answer_path = answer_root / f"{dataset}_{split}.jsonl"
        trajectory_path = trajectory_root / f"{dataset}_{split}.jsonl"
        if not answer_path.exists() or not trajectory_path.exists():
            raise SystemExit(f"{dataset}: answer/trajectory 파일이 없습니다: {split}")
        answer_rows = list(_iter_jsonl(answer_path))[:maximum]
        selected_ids = {str(row["id"]) for row in answer_rows}
        trajectory_rows = {
            str(row["id"]): row
            for row in _iter_jsonl(trajectory_path)
            if str(row["id"]) in selected_ids
        }
        if len(answer_rows) != len(trajectory_rows):
            raise SystemExit(f"{dataset}: answer/trajectory ID가 일치하지 않습니다.")
        for row in answer_rows:
            if tuple(row.get("actions", {})) != ACTIONS:
                raise SystemExit(f"{dataset}: 지원하지 않는 answer action 형식입니다: {row.get('id')}")
            trajectory = trajectory_rows[str(row["id"])]
            if tuple(trajectory.get("actions", {})) != ACTIONS:
                raise SystemExit(f"{dataset}: 지원하지 않는 trajectory action 형식입니다: {row.get('id')}")
            if len(row.get("planner_audit", [])) != MAX_SEARCH_STEPS:
                raise SystemExit(f"{dataset}: planner state 수가 맞지 않습니다: {row.get('id')}")
        rows.extend(answer_rows)
        trajectories.update(trajectory_rows)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("중복 질문 ID가 있습니다.")
    return rows, trajectories


def _normalise_query(value: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", value.casefold()).strip()


def _rule_observation(row: dict, trajectory: dict) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    initial_scores = sorted(
        (float(item.get("score", 0.0)) for item in trajectory["actions"][ACTIONS[0]].get("evidence", [])),
        reverse=True,
    )
    second_score = initial_scores[1] if len(initial_scores) > 1 else (initial_scores[0] if initial_scores else 0.0)
    scores = [second_score]
    hard_stop = []
    audits = row["planner_audit"]
    queries = [_normalise_query(str(audit.get("query", ""))) for audit in audits]
    for step in range(MAX_SEARCH_STEPS):
        state = audits[step].get("policy_state", {})
        if step:
            last_score = state.get("last_added_score")
            scores.append(float(last_score) if last_score is not None else -1.0)
        prior_queries = [query for query in queries[:step] if query]
        repeated = bool(prior_queries) and prior_queries[-1] in prior_queries[:-1]
        sufficient = state.get("evidence_sufficient") is True or audits[step].get("evidence_sufficient") is True
        no_new = int(state.get("consecutive_no_new", 0)) > 0
        missing_added_score = step > 0 and state.get("last_added_score") is None
        hard_stop.append(bool(sufficient or no_new or repeated or missing_added_score))
    return tuple(scores), tuple(hard_stop)


def _build_batch(rows: list[dict], trajectories: dict[str, dict]) -> RuleBatch:
    observations = [_rule_observation(row, trajectories[str(row["id"])]) for row in rows]
    return RuleBatch(
        rows=tuple(rows),
        scores=np.asarray([observation[0] for observation in observations], dtype="float64"),
        hard_stop=np.asarray([observation[1] for observation in observations], dtype="bool"),
        action_em=np.asarray(
            [[float(row["actions"][action]["em"]) for action in ACTIONS] for row in rows],
            dtype="float64",
        ),
        action_f1=np.asarray(
            [[float(row["actions"][action]["f1"]) for action in ACTIONS] for row in rows],
            dtype="float64",
        ),
        action_source_recall=np.asarray(
            [[float(row["actions"][action]["source_recall"]) for action in ACTIONS] for row in rows],
            dtype="float64",
        ),
        action_all_support_recall=np.asarray(
            [[float(row["actions"][action]["all_support_recall"]) for action in ACTIONS] for row in rows],
            dtype="float64",
        ),
        datasets=np.asarray([str(row["dataset"]) for row in rows], dtype="object"),
    )


def _subset(batch: RuleBatch, indices: np.ndarray) -> RuleBatch:
    return RuleBatch(
        rows=tuple(batch.rows[int(index)] for index in indices),
        scores=batch.scores[indices],
        hard_stop=batch.hard_stop[indices],
        action_em=batch.action_em[indices],
        action_f1=batch.action_f1[indices],
        action_source_recall=batch.action_source_recall[indices],
        action_all_support_recall=batch.action_all_support_recall[indices],
        datasets=batch.datasets[indices],
    )


def _simulate_steps(batch: RuleBatch, thresholds: tuple[float, ...]) -> np.ndarray:
    if len(thresholds) != MAX_SEARCH_STEPS:
        raise ValueError("단계별 규칙 threshold 수가 맞지 않습니다.")
    stop_steps = np.full(len(batch.rows), MAX_SEARCH_STEPS, dtype="int64")
    active = np.ones(len(batch.rows), dtype="bool")
    for step in range(MAX_SEARCH_STEPS):
        if step == 0:
            continue_search = batch.scores[:, step] < thresholds[step]
        else:
            continue_search = batch.scores[:, step] >= thresholds[step]
        continue_search &= ~batch.hard_stop[:, step]
        stopping = active & ~continue_search
        stop_steps[stopping] = step
        active &= continue_search
    return stop_steps


def _selected(values: np.ndarray, stop_steps: np.ndarray) -> np.ndarray:
    return values[np.arange(len(values)), stop_steps]


def _objective(
    batch: RuleBatch,
    stop_steps: np.ndarray,
    datasets: tuple[str, ...],
    f1_weight: float,
    em_weight: float,
    search_cost: float,
    harm_penalty: float,
) -> tuple[float, float, float]:
    selected_utility = f1_weight * _selected(batch.action_f1, stop_steps) + em_weight * _selected(
        batch.action_em, stop_steps
    )
    direct_utility = f1_weight * batch.action_f1[:, 0] + em_weight * batch.action_em[:, 0]
    policy_by_dataset = np.asarray(
        [float(selected_utility[batch.datasets == dataset].mean()) for dataset in datasets],
        dtype="float64",
    )
    direct_by_dataset = np.asarray(
        [float(direct_utility[batch.datasets == dataset].mean()) for dataset in datasets],
        dtype="float64",
    )
    balanced = float(np.mean(policy_by_dataset) - harm_penalty * np.maximum(0.0, direct_by_dataset - policy_by_dataset).sum())
    average_steps = float(stop_steps.mean())
    return balanced - search_cost * average_steps, balanced, average_steps


def _threshold_candidates(values: np.ndarray, steps: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.asarray([-1.0, 0.0, 1.0], dtype="float64")
    candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, steps)))
    return np.unique(
        np.concatenate(
            [
                np.asarray([np.nextafter(candidates[0], -np.inf)]),
                candidates,
                np.asarray([np.nextafter(candidates[-1], np.inf)]),
            ]
        )
    )


def _calibrate(batch: RuleBatch, args, grids: tuple[np.ndarray, ...]) -> tuple[float, ...]:
    def score(candidate: tuple[float, ...]) -> tuple[float, float, float, tuple[float, ...]]:
        objective, balanced, average_steps = _objective(
            batch,
            _simulate_steps(batch, candidate),
            args.datasets,
            args.f1_weight,
            args.em_weight,
            args.search_cost,
            args.dataset_harm_penalty,
        )
        return objective, balanced, -average_steps, candidate

    starts = (
        (float(grids[0][0]), float(grids[1][-1]), float(grids[2][-1])),
        tuple(float(grid[len(grid) // 2]) for grid in grids),
        (float(grids[0][-1]), float(grids[1][0]), float(grids[2][0])),
    )
    solutions = []
    for start in starts:
        current = start
        for _ in range(5):
            previous = current
            for step in range(MAX_SEARCH_STEPS):
                candidates = []
                for threshold in grids[step]:
                    candidate = list(current)
                    candidate[step] = float(threshold)
                    candidates.append(score(tuple(candidate)))
                current = max(candidates)[-1]
            if current == previous:
                break
        solutions.append(score(current))
    return max(solutions)[-1]


def _records(batch: RuleBatch, stop_steps: np.ndarray, thresholds: tuple[float, ...]) -> list[dict]:
    records = []
    for index, (row, stop_step) in enumerate(zip(batch.rows, stop_steps, strict=True)):
        action = ACTIONS[int(stop_step)]
        result = row["actions"][action]
        planner_latency = sum(
            float(item.get("latency_seconds", 0.0)) for item in row.get("planner_audit", [])[: int(stop_step)]
        )
        records.append(
            {
                "id": str(row["id"]),
                "dataset": str(row["dataset"]),
                "hop_count": int(row.get("hop_count", 0)),
                "selected_action": action,
                "stop_step": int(stop_step),
                "rule_scores": batch.scores[index].tolist(),
                "hard_stop": batch.hard_stop[index].tolist(),
                "thresholds": list(thresholds),
                "direct_em": float(row["direct_em"]),
                "direct_f1": float(row["direct_f1"]),
                "policy_em": float(result["em"]),
                "policy_f1": float(result["f1"]),
                "source_recall": float(result["source_recall"]),
                "all_support_recall": float(result["all_support_recall"]),
                "context_characters": int(result.get("context_characters", 0)),
                "reader_latency_seconds": float(result.get("reader_latency_seconds", 0.0)),
                "planner_latency_seconds": planner_latency,
                "oracle_f1": float(row["oracle_f1"]),
            }
        )
    return records


def _summary(records: list[dict], datasets: tuple[str, ...]) -> dict:
    summary = {}
    for dataset in datasets:
        selected = [record for record in records if record["dataset"] == dataset]
        direct_em = float(np.mean([record["direct_em"] for record in selected]))
        direct_f1 = float(np.mean([record["direct_f1"] for record in selected]))
        policy_em = float(np.mean([record["policy_em"] for record in selected]))
        policy_f1 = float(np.mean([record["policy_f1"] for record in selected]))
        step_counts = Counter(record["stop_step"] for record in selected)
        summary[dataset] = {
            "n": len(selected),
            "direct": {"em": direct_em, "f1": direct_f1},
            "policy": {"em": policy_em, "f1": policy_f1},
            "gain": {"em": policy_em - direct_em, "f1": policy_f1 - direct_f1},
            "average_extra_searches": float(np.mean([record["stop_step"] for record in selected])),
            "average_total_searches": 1.0 + float(np.mean([record["stop_step"] for record in selected])),
            "stop_steps": {
                str(step): {
                    "count": int(step_counts.get(step, 0)),
                    "rate": float(step_counts.get(step, 0) / len(selected)),
                }
                for step in range(MAX_SEARCH_STEPS + 1)
            },
            "source_recall": float(np.mean([record["source_recall"] for record in selected])),
            "all_support_recall": float(np.mean([record["all_support_recall"] for record in selected])),
            "average_context_characters": float(np.mean([record["context_characters"] for record in selected])),
            "average_reader_latency_seconds": float(
                np.mean([record["reader_latency_seconds"] for record in selected])
            ),
            "average_planner_latency_seconds": float(
                np.mean([record["planner_latency_seconds"] for record in selected])
            ),
            "oracle_f1": float(np.mean([record["oracle_f1"] for record in selected])),
            "oracle_regret_f1": float(np.mean([record["oracle_f1"] - record["policy_f1"] for record in selected])),
        }
    return summary


def _paired_bootstrap(records: list[dict], datasets: tuple[str, ...], samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    for dataset in datasets:
        selected = [record for record in records if record["dataset"] == dataset]
        dataset_result = {}
        for metric in ("em", "f1"):
            differences = np.asarray(
                [record[f"policy_{metric}"] - record[f"direct_{metric}"] for record in selected],
                dtype="float64",
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
        for record in sorted(records, key=lambda row: (row["dataset"], row["id"])):
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    started = perf_counter()
    train_rows, train_trajectories = _load_rows(
        Path(args.train_answer_root),
        Path(args.train_trajectory_root),
        args.datasets,
        "train",
        args.max_train_rows_per_dataset,
    )
    validation_rows, validation_trajectories = _load_rows(
        Path(args.validation_answer_root),
        Path(args.validation_trajectory_root),
        args.datasets,
        "validation",
        args.max_validation_rows_per_dataset,
    )
    train = _build_batch(train_rows, train_trajectories)
    validation = _build_batch(validation_rows, validation_trajectories)
    grids = tuple(_threshold_candidates(train.scores[:, step], args.threshold_steps) for step in range(MAX_SEARCH_STEPS))

    oof_records = []
    for fold in range(args.folds):
        test_indices = np.asarray(
            [
                index
                for index, row in enumerate(train.rows)
                if _bucket(str(row["id"]), "rule-stop-oof", args.folds) == fold
            ],
            dtype="int64",
        )
        test_index_set = set(test_indices.tolist())
        fit_indices = np.asarray(
            [index for index in range(len(train.rows)) if index not in test_index_set],
            dtype="int64",
        )
        fit, test = _subset(train, fit_indices), _subset(train, test_indices)
        thresholds = _calibrate(fit, args, grids)
        fold_records = _records(test, _simulate_steps(test, thresholds), thresholds)
        for record in fold_records:
            record["fold"] = fold
        oof_records.extend(fold_records)
        print(
            f"rule fold {fold + 1}/{args.folds}: fit/test={len(fit.rows):,}/{len(test.rows):,} "
            f"thresholds={'/'.join(f'{value:.4f}' for value in thresholds)}"
        )
    if len(oof_records) != len(train.rows) or len({record["id"] for record in oof_records}) != len(train.rows):
        raise AssertionError("규칙 OOF가 모든 train 질문을 정확히 한 번 평가하지 못했습니다.")

    thresholds = _calibrate(train, args, grids)
    validation_records = _records(validation, _simulate_steps(validation, thresholds), thresholds)
    result = {
        "format_version": FORMAT_VERSION,
        "rule": RULE_NAME,
        "rule_description": {
            "step_0": "continue when initial second-best source score is below threshold[0]",
            "step_1_2": "continue when last newly-added source score is at least the step threshold",
            "hard_stop": "planner sufficient, no-new evidence, missing added score, or repeated rewrite",
            "gold_at_inference": False,
            "hop_count_at_inference": False,
        },
        "datasets": list(args.datasets),
        "thresholds": list(thresholds),
        "calibration": {
            "split": "train",
            "rows": len(train.rows),
            "folds": args.folds,
            "threshold_steps": args.threshold_steps,
            "f1_weight": args.f1_weight,
            "em_weight": args.em_weight,
            "search_cost": args.search_cost,
            "dataset_harm_penalty": args.dataset_harm_penalty,
            "train_row_fingerprint": hashlib.sha256(
                "\n".join(sorted(str(row["id"]) for row in train.rows)).encode("utf-8")
            ).hexdigest(),
        },
        "train_oof_metrics": _summary(oof_records, args.datasets),
        "validation_metrics": _summary(validation_records, args.datasets),
        "paired_bootstrap_vs_direct": _paired_bootstrap(
            validation_records,
            args.datasets,
            args.bootstrap_samples,
            args.bootstrap_seed,
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "predictions": args.predictions_out,
        "train_oof_predictions": args.train_oof_predictions_out,
    }
    _write_jsonl(Path(args.predictions_out), validation_records)
    _write_jsonl(Path(args.train_oof_predictions_out), oof_records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"규칙 기반 STOP 평가 완료 · 총 소요 {_duration(perf_counter() - started)}")


if __name__ == "__main__":
    main()
