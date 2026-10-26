"""Evaluate a frozen evidence-conditioned Bridge policy on an unseen split.

The evaluator consumes already-cached action outcomes.  It never calls the
reader, fits a model, or calibrates a threshold on the evaluation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from experiments.evaluate_gated_policy import paired_bootstrap_summary, summarize_records
from experiments.train_bridge_policy import (
    ALL_ACTIONS,
    BridgePolicyModels,
    Thresholds,
    _load_rows,
    _predict,
    _select,
)
from experiments.train_multihop_policy import SUPPORTED_DATASETS


def main() -> None:
    args = _arguments()
    rows = _load_rows(
        Path(args.base_oracle_root),
        Path(args.bridge_oracle_root),
        args.datasets,
        args.split,
        args.max_rows_per_dataset,
    )
    payload = joblib.load(args.model)
    required = {"policy_type", "feature_names", "non_direct_actions", "thresholds", "gate", "value_model"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise SystemExit(f"지원하지 않는 Bridge 정책 모델입니다: {args.model}")
    if tuple(payload["non_direct_actions"]) != tuple(ALL_ACTIONS[1:]):
        raise SystemExit("정책 action space가 Bridge evaluator와 다릅니다.")
    if args.static_action not in ALL_ACTIONS:
        raise SystemExit(f"지원하지 않는 고정 baseline 행동입니다: {args.static_action}")

    models = BridgePolicyModels(payload["gate"], payload["value_model"])
    thresholds = Thresholds(
        gate=float(payload["thresholds"]["gate"]),
        advantage=float(payload["thresholds"]["advantage"]),
    )
    probabilities, advantages = _predict(models, rows)
    actions, em_values, f1_values = _select(rows, probabilities, advantages, thresholds)
    records = []
    for row, action, probability, estimates, em, f1 in zip(
        rows, actions, probabilities, advantages, em_values, f1_values, strict=True
    ):
        static = row["actions"][args.static_action]
        records.append({
            "id": row["id"], "dataset": row["dataset"], "split": args.split,
            "gate_probability": float(probability),
            "gate_threshold": thresholds.gate,
            "predicted_action": ALL_ACTIONS[1:][int(np.argmax(estimates))],
            "predicted_advantage": float(np.max(estimates)),
            "selected_action": action,
            "direct_em": row["direct_em"], "direct_f1": row["direct_f1"],
            "static_action": args.static_action,
            "static_em": float(static["em"]), "static_f1": float(static["f1"]),
            "policy_em": em, "policy_f1": f1,
            "oracle_em": row["oracle_em"], "oracle_f1": row["oracle_f1"],
        })

    metrics = summarize_records(records, args.datasets)
    result = {
        "split": args.split, "datasets": list(args.datasets), "rows": len(records),
        "model": args.model, "static_action": args.static_action,
        "metrics": metrics,
        "paired_bootstrap": {
            "policy_vs_direct": paired_bootstrap_summary(records, args.datasets, baseline="direct", samples=args.bootstrap_samples, seed=args.bootstrap_seed),
            "policy_vs_static": paired_bootstrap_summary(records, args.datasets, baseline="static", samples=args.bootstrap_samples, seed=args.bootstrap_seed + 2),
        },
    }
    _write_jsonl(Path(args.predictions_out), records)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for dataset in (*args.datasets, "Macro"):
        value = metrics[dataset]
        print(
            f"{dataset}: Direct EM/F1={value['direct_em']:.3f}/{value['direct_f1']:.3f} | "
            f"Policy={value['policy_em']:.3f}/{value['policy_f1']:.3f} | "
            f"ΔF1={value['f1_gain']:+.3f} | graph={value['graph_rate']:.1%}", flush=True
        )
    ci = result["paired_bootstrap"]["policy_vs_direct"]["f1"]
    print(f"Macro paired-bootstrap F1 gain={ci['observed_gain']:+.4f} 95% CI [{ci['ci95_low']:+.4f}, {ci['ci95_high']:+.4f}]", flush=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="동결 Bridge 정책의 unseen split 평가")
    parser.add_argument("--base-oracle-root", default="data/benchmarks/multihop/answer_oracle")
    parser.add_argument("--bridge-oracle-root", default="data/benchmarks/multihop/bridge_answer_oracle_1000")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument("--model", default="data/models/bridge_policy_full_1000.joblib")
    parser.add_argument("--static-action", default="bridge_path_1")
    parser.add_argument("--predictions-out", default="data/benchmarks/multihop/policy_eval/bridge_validation_predictions.jsonl")
    parser.add_argument("--summary-out", default="data/benchmarks/multihop/policy_eval/bridge_validation_summary.json")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_rows_per_dataset < 0 or args.bootstrap_samples < 0:
        parser.error("max rows와 bootstrap samples는 0 이상이어야 합니다.")
    return args


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for record in sorted(records, key=lambda row: (row["dataset"], row["id"])):
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
