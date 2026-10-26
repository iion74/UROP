"""Replay trusted published policy models on text-free numerical validation data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.evaluate_sequential_answer_gain_policy import _simulate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(42, 47), default=42)
    parser.add_argument("--replay-root", type=Path, default=ROOT / "artifacts/replay/full_validation")
    args = parser.parse_args()
    saved = joblib.load(ROOT / f"data/models/sequential_answer_gain_multiseed/policy_seed{args.seed}.joblib")
    report = json.loads((ROOT / "data/benchmarks/multihop/context_reader_aware/analysis/sequential_multiseed" /
                         f"policy_seed{args.seed}.json").read_text())
    results = {}
    for dataset in ("hotpotqa", "musique", "2wiki"):
        with np.load(args.replay_root / f"{dataset}.npz", allow_pickle=False) as arrays:
            if tuple(arrays["feature_names"]) != tuple(saved["feature_names"]):
                raise ValueError("Feature schema mismatch")
            states = arrays["x"]
            predictions = saved["model"].predict(states.reshape(-1, states.shape[-1])).reshape(-1, 3)
            steps = _simulate(predictions, saved["thresholds"])
            indices = np.arange(len(steps))
            results[dataset] = {
                "questions": len(steps),
                "f1": float(arrays["f1"][indices, steps].mean() * 100),
                "em": float(arrays["em"][indices, steps].mean() * 100),
                "average_total_searches": float(1 + steps.mean()),
                "reader_generated_tokens": float(arrays["reader_tokens"][indices, steps].mean()),
                "estimated_latency_seconds": float(arrays["estimated_latency"][indices, steps].mean()),
            }
        expected = report["evaluations"]["full_validation"][dataset]
        if results[dataset]["questions"] != expected["questions"]:
            raise ValueError(f"Validation question count mismatch: {dataset}")
        for key in results[dataset].keys() - {"questions"}:
            if not np.isclose(results[dataset][key], expected["policy"][key], atol=1e-10, rtol=0):
                raise ValueError(f"Published result does not match replay: {dataset}/{key}")
    print(json.dumps({"seed": args.seed, "matches_original_report": True, "datasets": results}, indent=2))


if __name__ == "__main__":
    main()
