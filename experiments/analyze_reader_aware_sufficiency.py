"""Paired analysis for the frozen reader-aware sufficiency candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reader-aware sufficiency final analysis")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--direct-root", default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_validation_2000")
    parser.add_argument("--short-proof-root", default="data/benchmarks/multihop/causal_evidence_reader/factorized_graph_repair_validation_2000")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--validation-offset", type=int, default=500)
    parser.add_argument("--rows-per-dataset", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(args.rows_per_dataset, args.bootstrap_samples) < 1 or args.validation_offset < 0:
        parser.error("rows/bootstrap은 1 이상이고 offset은 0 이상이어야 합니다.")
    return args


def _slice(path: Path, offset: int, count: int) -> list[dict]:
    rows = list(_iter_jsonl(path))[offset : offset + count]
    if len(rows) != count:
        raise SystemExit(f"요청한 분석 행 수가 없습니다: {path}")
    return rows


def _paired_interval(differences: np.ndarray, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype="float64")
    for start in range(0, samples, 256):
        end = min(samples, start + 256)
        indices = rng.integers(0, len(differences), size=(end - start, len(differences)))
        draws[start:end] = differences[indices].mean(axis=1) * 100.0
    return {
        "observed_change_points": float(differences.mean() * 100.0),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0.0)),
        "probability_noninferior_margin_1_point": float(np.mean(draws >= -1.0)),
    }


def main() -> None:
    args = _arguments()
    predictions = list(_iter_jsonl(Path(args.predictions)))
    result = {}
    for dataset_index, dataset in enumerate(args.datasets):
        adaptive_rows = [row for row in predictions if str(row["dataset"]) == dataset]
        if len(adaptive_rows) != args.rows_per_dataset:
            raise SystemExit(f"{dataset}: adaptive 예측 행 수가 맞지 않습니다.")
        adaptive_by_id = {str(row["id"]): row for row in adaptive_rows}
        direct_rows = _slice(
            Path(args.direct_root) / f"{dataset}_validation.jsonl",
            args.validation_offset,
            args.rows_per_dataset,
        )
        short_rows = _slice(
            Path(args.short_proof_root) / f"{dataset}_validation.jsonl",
            args.validation_offset,
            args.rows_per_dataset,
        )
        direct_by_id = {str(row["id"]): row for row in direct_rows}
        short_by_id = {str(row["id"]): row for row in short_rows}
        ids = [str(row["id"]) for row in adaptive_rows]
        if set(ids) != direct_by_id.keys() or set(ids) != short_by_id.keys():
            raise SystemExit(f"{dataset}: 비교군 ID가 adaptive와 일치하지 않습니다.")
        comparisons = {
            "adaptive_vs_chain_fixed3": {
                metric: _paired_interval(
                    np.asarray(
                        [
                            float(adaptive_by_id[row_id][metric])
                            - float(adaptive_by_id[row_id][f"fixed3_{metric}"])
                            for row_id in ids
                        ]
                    ),
                    args.bootstrap_samples,
                    args.seed + dataset_index * 20 + metric_index,
                )
                for metric_index, metric in enumerate(("em", "f1"))
            },
            "adaptive_vs_direct": {
                metric: _paired_interval(
                    np.asarray(
                        [
                            float(adaptive_by_id[row_id][metric])
                            - float(direct_by_id[row_id]["actions"]["vector"][metric])
                            for row_id in ids
                        ]
                    ),
                    args.bootstrap_samples,
                    args.seed + dataset_index * 20 + 2 + metric_index,
                )
                for metric_index, metric in enumerate(("em", "f1"))
            },
            "chain_fixed3_vs_short_proof_fixed3": {
                metric: _paired_interval(
                    np.asarray(
                        [
                            float(adaptive_by_id[row_id][f"fixed3_{metric}"])
                            - float(short_by_id[row_id][metric])
                            for row_id in ids
                        ]
                    ),
                    args.bootstrap_samples,
                    args.seed + dataset_index * 20 + 4 + metric_index,
                )
                for metric_index, metric in enumerate(("em", "f1"))
            },
        }
        average_searches = float(np.mean([row["chosen_step"] for row in adaptive_rows]))
        result[dataset] = {
            "count": len(ids),
            "direct": {
                "em": float(np.mean([direct_by_id[row_id]["actions"]["vector"]["em"] for row_id in ids])),
                "f1": float(np.mean([direct_by_id[row_id]["actions"]["vector"]["f1"] for row_id in ids])),
                "average_additional_searches": 0.0,
            },
            "short_proof_fixed3": {
                "em": float(np.mean([short_by_id[row_id]["em"] for row_id in ids])),
                "f1": float(np.mean([short_by_id[row_id]["f1"] for row_id in ids])),
                "average_additional_searches": 3.0,
            },
            "chain_proof_fixed3": {
                "em": float(np.mean([adaptive_by_id[row_id]["fixed3_em"] for row_id in ids])),
                "f1": float(np.mean([adaptive_by_id[row_id]["fixed3_f1"] for row_id in ids])),
                "average_additional_searches": 3.0,
            },
            "reader_aware_adaptive": {
                "em": float(np.mean([adaptive_by_id[row_id]["em"] for row_id in ids])),
                "f1": float(np.mean([adaptive_by_id[row_id]["f1"] for row_id in ids])),
                "average_additional_searches": average_searches,
                "search_reduction_percent_vs_fixed3": (3.0 - average_searches) / 3.0 * 100.0,
            },
            "comparisons": comparisons,
        }
    payload = {
        "format_version": 1,
        "method": "reader_aware_counterfactual_sufficiency_final_heldout",
        "validation_offset": args.validation_offset,
        "rows_per_dataset": args.rows_per_dataset,
        "bootstrap_samples": args.bootstrap_samples,
        "datasets": result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
