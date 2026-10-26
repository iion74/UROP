"""Aggregate repeated 1,000-question validation cohorts from a frozen 2,000 pool.

The retrieval and reader are evaluated once for every question in the fixed
corpus.  This script then reproduces the benchmark's stable-hash sampler for
multiple query seeds, avoiding another LLM call while changing the actual
question sample.  Results are always reported per dataset; no cross-dataset
Macro score is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from itertools import combinations
from pathlib import Path

from experiments.train_multihop_policy import SUPPORTED_DATASETS, _dataset_seed, _iter_jsonl


ACTIONS = ("vector", "planned_continue_1", "planned_continue_2", "planned_continue_3")
DEFAULT_COUNTS = {
    "hotpotqa": {2: 1000},
    "musique": {2: 400, 3: 350, 4: 250},
    "2wiki": {2: 1000},
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated stable-hash validation analysis")
    parser.add_argument("--old-answer-root", required=True)
    parser.add_argument("--new-answer-root", required=True)
    parser.add_argument("--old-trajectory-root", required=True)
    parser.add_argument("--new-trajectory-root", required=True)
    parser.add_argument("--policy-predictions", nargs="*", metavar="NAME=PATH", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.seeds = tuple(dict.fromkeys(args.seeds))
    if len(args.seeds) < 2:
        parser.error("반복 검증에는 서로 다른 seed가 2개 이상 필요합니다.")
    return args


def _load(root: str, dataset: str, suffix: str = "validation") -> dict[str, dict]:
    path = Path(root) / f"{dataset}_{suffix}.jsonl"
    rows = {str(row["id"]): row for row in _iter_jsonl(path)}
    if len(rows) < 1000:
        raise SystemExit(f"{path}: 반복 표본을 만들기에 행이 부족합니다: {len(rows):,}")
    return rows


def _load_policy(spec: str) -> tuple[str, dict[str, dict]]:
    try:
        name, raw_path = spec.split("=", 1)
    except ValueError as error:
        raise SystemExit(f"policy prediction 형식은 NAME=PATH입니다: {spec}") from error
    rows = {str(row["id"]): row for row in _iter_jsonl(Path(raw_path))}
    return name, rows


def _sample_ids(rows: dict[str, dict], dataset: str, seed: int) -> tuple[str, ...]:
    selected: list[str] = []
    dataset_seed = _dataset_seed(seed, dataset)
    for hop, requested in DEFAULT_COUNTS[dataset].items():
        available = [row_id for row_id, row in rows.items() if int(row.get("hop_count", 0)) == hop]
        ranked = sorted(
            available,
            key=lambda row_id: (
                hashlib.sha256(f"{dataset_seed}\0hop={hop}\0{row_id}".encode("utf-8")).digest(),
                row_id,
            ),
        )
        if len(ranked) < requested:
            raise SystemExit(f"{dataset} hop={hop}: {requested:,}개 표본을 만들 수 없습니다.")
        selected.extend(ranked[:requested])
    return tuple(sorted(selected))


def _mean(rows: list[dict], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return statistics.fmean(values)


def _action_metrics(rows: list[dict], action: str) -> dict[str, float]:
    return {
        "em": _mean(rows, ("actions", action, "em")),
        "f1": _mean(rows, ("actions", action, "f1")),
    }


def _retrieval_metrics(rows: list[dict], action: str) -> dict[str, float]:
    return {
        "source_recall": _mean(rows, ("actions", action, "source_recall")),
        "all_support_recall": _mean(rows, ("actions", action, "all_support_recall")),
    }


def _aggregate(per_seed: dict[str, dict]) -> dict:
    def walk(values: list, path: tuple[str, ...] = ()):
        first = values[0]
        if isinstance(first, dict):
            return {key: walk([value[key] for value in values], (*path, key)) for key in first}
        numeric = [float(value) for value in values]
        return {
            "mean": statistics.fmean(numeric),
            "std": statistics.stdev(numeric),
            "min": min(numeric),
            "max": max(numeric),
        }

    return walk(list(per_seed.values()))


def _overlap(samples: dict[str, tuple[str, ...]]) -> dict[str, float]:
    ratios = [
        len(set(samples[left]) & set(samples[right])) / len(samples[left])
        for left, right in combinations(samples, 2)
    ]
    return {
        "mean_pairwise_fraction": statistics.fmean(ratios),
        "min_pairwise_fraction": min(ratios),
        "max_pairwise_fraction": max(ratios),
        "union_questions": len(set().union(*(set(values) for values in samples.values()))),
    }


def main() -> None:
    args = _arguments()
    policies = dict(_load_policy(spec) for spec in args.policy_predictions)
    result = {
        "seeds": list(args.seeds),
        "sample_counts": {dataset: {str(k): v for k, v in counts.items()} for dataset, counts in DEFAULT_COUNTS.items()},
        "datasets": {},
    }
    for dataset in SUPPORTED_DATASETS:
        old_answers = _load(args.old_answer_root, dataset)
        new_answers = _load(args.new_answer_root, dataset)
        old_trajectories = _load(args.old_trajectory_root, dataset)
        new_trajectories = _load(args.new_trajectory_root, dataset)
        common = set(old_answers) & set(new_answers) & set(old_trajectories) & set(new_trajectories)
        if len(common) < 1000:
            raise SystemExit(f"{dataset}: old/new 공통 질문이 부족합니다: {len(common):,}")
        sampling_rows = {row_id: new_answers[row_id] for row_id in common}
        samples = {
            str(seed): _sample_ids(sampling_rows, dataset, seed)
            for seed in args.seeds
        }
        per_seed = {}
        for seed_text, ids in samples.items():
            old_answer_rows = [old_answers[row_id] for row_id in ids]
            new_answer_rows = [new_answers[row_id] for row_id in ids]
            old_retrieval_rows = [old_trajectories[row_id] for row_id in ids]
            new_retrieval_rows = [new_trajectories[row_id] for row_id in ids]
            old_c3 = _action_metrics(old_answer_rows, ACTIONS[-1])
            new_c3 = _action_metrics(new_answer_rows, ACTIONS[-1])
            old_retrieval = _retrieval_metrics(old_retrieval_rows, ACTIONS[-1])
            new_retrieval = _retrieval_metrics(new_retrieval_rows, ACTIONS[-1])
            entry = {
                "old_c3": old_c3,
                "new_c3": new_c3,
                "new_minus_old": {key: new_c3[key] - old_c3[key] for key in old_c3},
                "old_retrieval_c3": old_retrieval,
                "new_retrieval_c3": new_retrieval,
                "retrieval_new_minus_old": {
                    key: new_retrieval[key] - old_retrieval[key] for key in old_retrieval
                },
                "new_prefix": {action: _action_metrics(new_answer_rows, action) for action in ACTIONS},
            }
            if policies:
                entry["policies"] = {}
                for name, policy_rows in policies.items():
                    if not set(ids) <= set(policy_rows):
                        raise SystemExit(f"{dataset}: policy {name}에 반복 표본 ID가 없습니다.")
                    selected = [policy_rows[row_id] for row_id in ids]
                    entry["policies"][name] = {
                        "em": _mean(selected, ("policy_em",)),
                        "f1": _mean(selected, ("policy_f1",)),
                        "average_extra_searches": _mean(selected, ("stop_step",)),
                        "f1_minus_fixed_c3": _mean(selected, ("policy_f1",)) - new_c3["f1"],
                    }
            per_seed[seed_text] = entry
        result["datasets"][dataset] = {
            "common_pool": len(common),
            "sample_overlap": _overlap(samples),
            "per_seed": per_seed,
            "aggregate": _aggregate(per_seed),
        }
        if policies:
            by_model = {
                name: _aggregate({seed: entry["policies"][name] for seed, entry in per_seed.items()})
                for name in policies
            }
            by_sample = {
                seed: _aggregate(entry["policies"])
                for seed, entry in per_seed.items()
            }
            all_runs = {
                f"sample={seed},model={name}": metrics
                for seed, entry in per_seed.items()
                for name, metrics in entry["policies"].items()
            }
            result["datasets"][dataset]["policy_robustness"] = {
                "query_samples_per_model": by_model,
                "model_seeds_per_query_sample": by_sample,
                "all_query_and_model_seed_runs": _aggregate(all_runs),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
