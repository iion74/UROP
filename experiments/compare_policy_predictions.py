"""Paired, per-dataset comparison of two adaptive STOP policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare adaptive policy predictions by question")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--search-cost", type=float, default=0.002)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.search_cost < 0.0 or args.bootstrap_samples < 1:
        parser.error("검색 비용은 0 이상이고 bootstrap samples는 1 이상이어야 합니다.")
    return args


def _load_pair(reference_path: Path, candidate_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    reference = {str(row["id"]): row for row in _iter_jsonl(reference_path)}
    candidate = {str(row["id"]): row for row in _iter_jsonl(candidate_path)}
    if not reference or reference.keys() != candidate.keys():
        raise SystemExit("정책 예측 ID 집합이 일치하지 않습니다.")
    return reference, candidate


def _paired_metric(differences: np.ndarray, samples: int, rng: np.random.Generator) -> dict:
    draws = np.empty(samples, dtype="float64")
    for start in range(0, samples, 256):
        end = min(samples, start + 256)
        indices = rng.integers(0, len(differences), size=(end - start, len(differences)))
        draws[start:end] = differences[indices].mean(axis=1)
    return {
        "observed_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0.0)),
    }


def _compare(
    reference: dict[str, dict],
    candidate: dict[str, dict],
    datasets: tuple[str, ...],
    samples: int,
    seed: int,
    search_cost: float,
) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    for dataset in datasets:
        ids = sorted(row_id for row_id, row in reference.items() if str(row["dataset"]) == dataset)
        if not ids:
            raise ValueError(f"{dataset}: 비교할 예측이 없습니다.")
        reference_rows = [reference[row_id] for row_id in ids]
        candidate_rows = [candidate[row_id] for row_id in ids]
        dataset_result = {"n": len(ids), "reference": {}, "candidate": {}, "candidate_minus_reference": {}}
        for metric in ("em", "f1"):
            reference_values = np.asarray([float(row[f"policy_{metric}"]) for row in reference_rows])
            candidate_values = np.asarray([float(row[f"policy_{metric}"]) for row in candidate_rows])
            differences = candidate_values - reference_values
            dataset_result["reference"][metric] = float(reference_values.mean())
            dataset_result["candidate"][metric] = float(candidate_values.mean())
            dataset_result["candidate_minus_reference"][metric] = _paired_metric(differences, samples, rng)
            dataset_result["candidate_minus_reference"][metric]["win_rate"] = float(np.mean(differences > 0.0))
            dataset_result["candidate_minus_reference"][metric]["tie_rate"] = float(np.mean(differences == 0.0))
            dataset_result["candidate_minus_reference"][metric]["loss_rate"] = float(np.mean(differences < 0.0))
        reference_steps = np.asarray([int(row["stop_step"]) for row in reference_rows], dtype="float64")
        candidate_steps = np.asarray([int(row["stop_step"]) for row in candidate_rows], dtype="float64")
        dataset_result["reference"]["average_extra_searches"] = float(reference_steps.mean())
        dataset_result["candidate"]["average_extra_searches"] = float(candidate_steps.mean())
        dataset_result["candidate_minus_reference"]["average_extra_searches"] = float(
            (candidate_steps - reference_steps).mean()
        )
        reference_utility = np.asarray(
            [0.5 * float(row["policy_em"]) + 0.5 * float(row["policy_f1"]) for row in reference_rows]
        ) - search_cost * reference_steps
        candidate_utility = np.asarray(
            [0.5 * float(row["policy_em"]) + 0.5 * float(row["policy_f1"]) for row in candidate_rows]
        ) - search_cost * candidate_steps
        dataset_result["candidate_minus_reference"]["cost_adjusted_utility"] = _paired_metric(
            candidate_utility - reference_utility,
            samples,
            rng,
        )
        result[dataset] = dataset_result
    return result


def main() -> None:
    args = _arguments()
    reference, candidate = _load_pair(Path(args.reference), Path(args.candidate))
    result = {
        "reference": {"name": args.reference_name, "path": args.reference},
        "candidate": {"name": args.candidate_name, "path": args.candidate},
        "datasets": _compare(
            reference,
            candidate,
            args.datasets,
            args.bootstrap_samples,
            args.bootstrap_seed,
            args.search_cost,
        ),
        "search_cost": args.search_cost,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
