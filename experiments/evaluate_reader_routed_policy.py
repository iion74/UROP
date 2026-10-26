"""Merge a frozen STOP policy with a validation-gated reader prompt.

The STOP policy is never recalibrated on validation answers.  A prompt variant
can be enabled per domain only after a separate gate experiment; rows that stop
before the variant's scored action keep the baseline reader prediction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl


CHAIN_ACTION = "planned_continue_3"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STOP policy + gated evidence-chain reader 평가")
    parser.add_argument(
        "--policy-predictions",
        default=(
            "data/benchmarks/multihop/policy_eval/"
            "stop_value_ensemble5_cost005_validation_2000_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--baseline-answer-root",
        default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_validation_2000",
    )
    parser.add_argument(
        "--chain-answer-root",
        default="data/benchmarks/multihop/reader_eval/evidence_chain_v3_validation_2000",
    )
    parser.add_argument(
        "--chain-datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=["musique", "2wiki"],
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default="data/benchmarks/multihop/policy_eval/gated_reader_policy.json")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--gate-rows-per-chain-dataset",
        type=int,
        default=200,
        help="prompt 확대 여부 판단에 사용한 각 chain dataset의 선두 행 수",
    )
    args = parser.parse_args()
    args.chain_datasets = tuple(dict.fromkeys(args.chain_datasets))
    if args.bootstrap_samples < 1 or args.gate_rows_per_chain_dataset < 0:
        parser.error("bootstrap samples는 1 이상, gate rows는 0 이상이어야 합니다.")
    return args


def _load_answer_map(root: Path, datasets: tuple[str, ...], split: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for dataset in datasets:
        path = root / f"{dataset}_{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"answer 결과가 없습니다: {path}")
        for row in _iter_jsonl(path):
            row_id = str(row["id"])
            if row_id in rows:
                raise SystemExit(f"중복 answer ID입니다: {row_id}")
            rows[row_id] = row
    return rows


def _gate_ids(root: Path, datasets: tuple[str, ...], split: str, count: int) -> set[str]:
    ids: set[str] = set()
    if count == 0:
        return ids
    for dataset in datasets:
        path = root / f"{dataset}_{split}.jsonl"
        selected = list(_iter_jsonl(path))[:count]
        if len(selected) < count:
            raise SystemExit(f"{dataset}: gate rows {count:,}개보다 chain answer가 적습니다.")
        ids.update(str(row["id"]) for row in selected)
    return ids


def _route_records(
    policy_rows: list[dict],
    baseline_by_id: dict[str, dict],
    chain_by_id: dict[str, dict],
    chain_datasets: tuple[str, ...],
) -> list[dict]:
    records: list[dict] = []
    for policy in policy_rows:
        row_id = str(policy["id"])
        dataset = str(policy["dataset"])
        action = str(policy["selected_action"])
        baseline = baseline_by_id.get(row_id)
        if baseline is None:
            raise SystemExit(f"baseline answer ID가 없습니다: {row_id}")
        use_chain = dataset in chain_datasets and action == CHAIN_ACTION
        answer_row = chain_by_id.get(row_id) if use_chain else baseline
        if answer_row is None:
            raise SystemExit(f"evidence-chain answer ID가 없습니다: {row_id}")
        selected = answer_row.get("actions", {}).get(action)
        if not isinstance(selected, dict):
            raise SystemExit(f"{row_id}: 선택 action answer가 없습니다: {action}")
        direct = baseline["actions"]["vector"]
        records.append(
            {
                **policy,
                "baseline_policy_em": float(policy["policy_em"]),
                "baseline_policy_f1": float(policy["policy_f1"]),
                "policy_em": float(selected["em"]),
                "policy_f1": float(selected["f1"]),
                "direct_em": float(direct["em"]),
                "direct_f1": float(direct["f1"]),
                "reader_route": "evidence_chain_v3" if use_chain else "baseline_v2",
            }
        )
    return records


def _metrics(records: list[dict], datasets: tuple[str, ...]) -> dict:
    result = {}
    for dataset in datasets:
        selected = [row for row in records if row["dataset"] == dataset]
        result[dataset] = {
            "n": len(selected),
            "em": float(np.mean([row["policy_em"] for row in selected])),
            "f1": float(np.mean([row["policy_f1"] for row in selected])),
            "baseline_policy_em": float(np.mean([row["baseline_policy_em"] for row in selected])),
            "baseline_policy_f1": float(np.mean([row["baseline_policy_f1"] for row in selected])),
            "direct_em": float(np.mean([row["direct_em"] for row in selected])),
            "direct_f1": float(np.mean([row["direct_f1"] for row in selected])),
            "average_extra_searches": float(np.mean([row["stop_step"] for row in selected])),
            "reader_routes": dict(sorted(Counter(row["reader_route"] for row in selected).items())),
        }
    return result


def _paired_bootstrap(
    records: list[dict],
    datasets: tuple[str, ...],
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    comparisons = {
        "vs_direct": ("direct_em", "direct_f1"),
        "vs_baseline_policy": ("baseline_policy_em", "baseline_policy_f1"),
    }
    for dataset in datasets:
        selected = [row for row in records if row["dataset"] == dataset]
        dataset_result = {}
        for name, (em_reference, f1_reference) in comparisons.items():
            comparison = {}
            for metric, reference in (("em", em_reference), ("f1", f1_reference)):
                differences = np.asarray(
                    [float(row[f"policy_{metric}"]) - float(row[reference]) for row in selected],
                    dtype="float64",
                )
                draws = np.empty(samples, dtype="float64")
                for start in range(0, samples, 256):
                    end = min(samples, start + 256)
                    indices = rng.integers(0, len(differences), size=(end - start, len(differences)))
                    draws[start:end] = differences[indices].mean(axis=1)
                comparison[metric] = {
                    "observed_gain": float(differences.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "probability_positive": float(np.mean(draws > 0.0)),
                }
            dataset_result[name] = comparison
        result[dataset] = dataset_result
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in sorted(rows, key=lambda item: (item["dataset"], item["id"])):
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = _arguments()
    policy_rows = list(_iter_jsonl(Path(args.policy_predictions)))
    datasets = tuple(dataset for dataset in SUPPORTED_DATASETS if any(row["dataset"] == dataset for row in policy_rows))
    baseline = _load_answer_map(Path(args.baseline_answer_root), datasets, args.split)
    chain = _load_answer_map(Path(args.chain_answer_root), args.chain_datasets, args.split)
    records = _route_records(policy_rows, baseline, chain, args.chain_datasets)
    gate_ids = _gate_ids(
        Path(args.chain_answer_root),
        args.chain_datasets,
        args.split,
        args.gate_rows_per_chain_dataset,
    )
    held_out_records = [
        row for row in records if row["dataset"] in args.chain_datasets and row["id"] not in gate_ids
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output.with_name(f"{output.stem}_predictions.jsonl")
    _write_jsonl(prediction_path, records)
    payload = {
        "format_version": 1,
        "policy": "five_seed_stop_vote_with_validation_gated_reader",
        "policy_predictions": str(Path(args.policy_predictions)),
        "chain_datasets": list(args.chain_datasets),
        "chain_action": CHAIN_ACTION,
        "gate_rows_per_chain_dataset": args.gate_rows_per_chain_dataset,
        "validation": _metrics(records, datasets),
        "paired_bootstrap": _paired_bootstrap(
            records, datasets, args.bootstrap_samples, args.bootstrap_seed
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "held_out_after_gate": {
            "validation": _metrics(held_out_records, args.chain_datasets),
            "paired_bootstrap": _paired_bootstrap(
                held_out_records,
                args.chain_datasets,
                args.bootstrap_samples,
                args.bootstrap_seed + 1,
            ),
        },
        "predictions": str(prediction_path),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
