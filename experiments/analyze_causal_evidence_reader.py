"""Summarize paired causal-evidence reader results without pooling datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze causal evidence reader outputs")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--external-baseline-root",
        help="Optional result root whose baseline_em/baseline_f1 fields represent Direct RAG.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=list(SUPPORTED_DATASETS),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--design-prefix", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.design_prefix < 0 or args.bootstrap_samples < 1:
        parser.error("design prefix는 0 이상, bootstrap samples는 1 이상이어야 합니다.")
    return args


def _paired_interval(differences: np.ndarray, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype="float64")
    for start in range(0, samples, 1_000):
        size = min(1_000, samples - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        draws[start : start + size] = differences[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "gain": float(differences.mean() * 100.0),
        "ci95_low": float(low * 100.0),
        "ci95_high": float(high * 100.0),
        "probability_positive": float(np.mean(draws > 0.0) * 100.0),
    }


def _means(rows: list[dict]) -> dict:
    def effective_width(row: dict) -> int:
        if "selected_indices" in row:
            return len(row["selected_indices"])
        return len(row.get("base_selected_indices", ())) + int(row.get("repair_count", 0))

    def predicted_width(row: dict) -> int:
        if "selected_width" in row:
            return int(row["selected_width"])
        return len(row.get("base_selected_indices", ()))

    return {
        "count": len(rows),
        "baseline_em": float(np.mean([row["baseline_em"] for row in rows]) * 100.0),
        "method_em": float(np.mean([row["em"] for row in rows]) * 100.0),
        "baseline_f1": float(np.mean([row["baseline_f1"] for row in rows]) * 100.0),
        "method_f1": float(np.mean([row["f1"] for row in rows]) * 100.0),
        "average_selected_width": float(np.mean([effective_width(row) for row in rows])),
        "predicted_width_distribution": dict(
            sorted(Counter(str(predicted_width(row)) for row in rows).items())
        ),
        "effective_width_distribution": dict(
            sorted(Counter(str(effective_width(row)) for row in rows).items())
        ),
    }


def _summarize(rows: list[dict], samples: int, seed: int) -> dict:
    result = _means(rows)
    result["paired_bootstrap"] = {
        "em": _paired_interval(
            np.asarray([row["em"] - row["baseline_em"] for row in rows]), samples, seed
        ),
        "f1": _paired_interval(
            np.asarray([row["f1"] - row["baseline_f1"] for row in rows]), samples, seed + 1
        ),
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("reasoning_type") or "unknown")].append(row)
    result["by_reasoning_type"] = {
        key: _means(value) for key, value in sorted(grouped.items())
    }
    return result


def main() -> None:
    args = _arguments()
    root = Path(args.input_root)
    report = {
        "input_root": str(root),
        "design_prefix": args.design_prefix,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "datasets": {},
    }
    for offset, dataset in enumerate(args.datasets):
        rows = list(_iter_jsonl(root / f"{dataset}_{args.split}.jsonl"))
        if not rows:
            raise SystemExit(f"결과가 없습니다: {dataset}")
        heldout = rows[args.design_prefix :]
        if not heldout:
            raise SystemExit(f"design prefix 뒤의 문항이 없습니다: {dataset}")
        report["datasets"][dataset] = {
            "all_rows": _summarize(rows, args.bootstrap_samples, args.bootstrap_seed + offset * 10),
            "heldout_after_design_prefix": _summarize(
                heldout, args.bootstrap_samples, args.bootstrap_seed + offset * 10 + 2
            ),
        }
        if args.external_baseline_root:
            external = {
                str(row["id"]): row
                for row in _iter_jsonl(
                    Path(args.external_baseline_root) / f"{dataset}_{args.split}.jsonl"
                )
            }
            comparison = []
            for row in rows:
                source = external.get(str(row["id"]))
                if source is None:
                    raise SystemExit(f"external baseline ID가 없습니다: {dataset}/{row['id']}")
                comparison.append(
                    {
                        **row,
                        "baseline_em": float(source["baseline_em"]),
                        "baseline_f1": float(source["baseline_f1"]),
                    }
                )
            report["datasets"][dataset]["all_rows_vs_external_baseline"] = _summarize(
                comparison, args.bootstrap_samples, args.bootstrap_seed + offset * 10 + 4
            )
            report["datasets"][dataset]["heldout_vs_external_baseline"] = _summarize(
                comparison[args.design_prefix :],
                args.bootstrap_samples,
                args.bootstrap_seed + offset * 10 + 6,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for dataset, result in report["datasets"].items():
        for cohort, summary in result.items():
            f1 = summary["paired_bootstrap"]["f1"]
            print(
                f"{dataset} {cohort}: n={summary['count']} "
                f"EM={summary['baseline_em']:.2f}->{summary['method_em']:.2f} "
                f"F1={summary['baseline_f1']:.2f}->{summary['method_f1']:.2f} "
                f"gain={f1['gain']:+.2f} CI=[{f1['ci95_low']:+.2f},{f1['ci95_high']:+.2f}] "
                f"P(gain>0)={f1['probability_positive']:.2f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
