"""Run five independent train-sample-seed policy experiments sequentially.

Prefix generation is resumable; rerunning this command continues incomplete
seeds and skips completed reports. No two experiment jobs run concurrently.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiments.analyze_sequential_answer_gain_multiseed import EXPECTED_COUNTS
from experiments.evaluate_answer_gain_search_policy import DATASETS, ROOT


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-root", type=Path, default=ROOT / "analysis" / "sequential_multiseed")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Specify at least three distinct training-sample seeds")
    return args


def _ready(root: Path, split: str, seed: int | None, expected: int) -> bool:
    for dataset in DATASETS:
        manifest_path = root / f"{dataset}_{split}.manifest.json"
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "ready" or
                manifest.get("completed_rows") != expected or
                (seed is not None and manifest.get("query_seed") != seed)):
            return False
    return True


def _valid_report(path: Path, seed: int, root: Path) -> bool:
    if not path.exists():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    return (
        report.get("training_sample_seed") == seed and
        report.get("training_root") == str(root) and
        report.get("training_questions") == 6000 and
        all(report.get("evaluations", {}).get("full_validation", {}).get(dataset, {}).get("questions") == count
            for dataset, count in EXPECTED_COUNTS.items())
    )


def _run(*args: str) -> None:
    command = [sys.executable, "-m", *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = _arguments()
    for dataset, count in EXPECTED_COUNTS.items():
        path = ROOT / "full_validation" / f"{dataset}_validation.manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "ready" or manifest.get("completed_rows") != count:
            raise ValueError(f"Incomplete validation prefixes: {path}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports: list[Path] = []
    try:
        for seed in args.seeds:
            train_root = ROOT / f"train_pool_seed{seed}_2000"
            if not _ready(train_root, "train", seed, 2000):
                _run(
                    "experiments.evaluate_context_reader_aware_prefixes",
                    "--datasets", *DATASETS,
                    "--split", "train",
                    "--max-queries-per-dataset", "2000",
                    "--query-seed", str(seed),
                    "--output-root", str(train_root),
                    "--log-every", str(args.log_every),
                )
            report = args.output_root / f"policy_seed{seed}.json"
            model = Path("data/models/sequential_answer_gain_multiseed") / f"policy_seed{seed}.joblib"
            if not _valid_report(report, seed, train_root):
                _run(
                    "experiments.evaluate_sequential_answer_gain_policy",
                    "--train-root", str(train_root),
                    "--validation-root", str(ROOT / "full_validation"),
                    "--skip-holdout",
                    "--output", str(report),
                    "--model-out", str(model),
                    "--search-cost", "0.001",
                    "--bootstrap-samples", "2000",
                )
            reports.append(report)
        _run(
            "experiments.analyze_sequential_answer_gain_multiseed",
            "--reports", *(str(path) for path in reports),
            "--output", str(ROOT / "analysis" / "sequential_answer_gain_multiseed.json"),
        )
    finally:
        # This job starts the frozen reader; unload it on success or failure.
        subprocess.run(["ollama", "stop", "qwen3:8b"], check=False)


if __name__ == "__main__":
    main()
