"""Finish the three-dataset, frozen-policy holdout sequentially and unload Qwen."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("data/benchmarks/multihop/context_reader_aware")
PREFIX = ROOT / "disjoint_holdout_seed20260916_500"
ANALYSIS = ROOT / "analysis" / "disjoint_holdout_seed20260916_500"
EXCLUDE = ROOT / "train_pool_seed42_2000"
DATASETS = ("hotpotqa", "musique", "2wiki")
MODEL = "qwen3:8b"


def _run(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _ready(dataset: str) -> bool:
    path = PREFIX / f"{dataset}_train.manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        manifest.get("status") == "ready"
        and manifest.get("requested_rows") == 500
        and manifest.get("completed_rows") == 500
        and manifest.get("query_seed") == 20260916
        and manifest.get("exclude_ids_root") == str(EXCLUDE)
    )


def main() -> None:
    print("HotpotQA 선행 작업 완료 대기; 이후 MuSiQue → 2Wiki → 동결 정책 평가", flush=True)
    try:
        started = time.monotonic()
        while not _ready("hotpotqa"):
            if time.monotonic() - started > 7200:
                raise TimeoutError("HotpotQA prefix가 2시간 안에 완료되지 않았습니다.")
            time.sleep(30)
        for dataset in DATASETS[1:]:
            if not _ready(dataset):
                _run(
                    "experiments.evaluate_context_reader_aware_prefixes",
                    "--datasets", dataset,
                    "--split", "train",
                    "--max-queries-per-dataset", "500",
                    "--query-seed", "20260916",
                    "--exclude-ids-root", str(EXCLUDE),
                    "--output-root", str(PREFIX),
                    "--ollama-keep-alive", "30m",
                    "--log-every", "25",
                )
        _run(
            "experiments.evaluate_frozen_dual_value_router",
            "--prefix-root", str(PREFIX),
            "--split", "train",
            "--output-root", str(ANALYSIS),
        )
        _run(
            "experiments.evaluate_dual_value_reasoning_pilot",
            "--split", "train",
            "--prefix-root", str(PREFIX),
            "--reason-queue", str(ANALYSIS / "frozen_router_train.reason_queue.jsonl"),
            "--rows-per-dataset", "0",
            "--output-root", str(ANALYSIS / "reason"),
            "--ollama-keep-alive", "30m",
            "--log-every", "10",
        )
        _run(
            "experiments.evaluate_dual_value_choice_judge",
            "--split", "train",
            "--prefix-root", str(PREFIX),
            "--pilot-root", str(ANALYSIS / "reason"),
            "--output-root", str(ANALYSIS / "judge"),
            "--ollama-keep-alive", "30m",
            "--log-every", "10",
        )
        full = ANALYSIS / "full_system.json"
        _run(
            "experiments.analyze_dual_value_full_system",
            "--prefix-root", str(PREFIX),
            "--split", "train",
            "--router-predictions", str(ANALYSIS / "frozen_router_train.predictions.jsonl"),
            "--judge-root", str(ANALYSIS / "judge"),
            "--output", str(full),
        )
        _run(
            "experiments.analyze_kci_required_baselines",
            "--prefix-root", str(PREFIX),
            "--split", "train",
            "--full-predictions", str(ANALYSIS / "full_system.predictions.jsonl"),
            "--output", str(ANALYSIS / "kci_baselines.json"),
        )
        print(f"holdout 완료: {full}", flush=True)
    finally:
        print("Qwen3:8B GPU 사용 종료", flush=True)
        subprocess.run(["ollama", "stop", MODEL], check=False)


if __name__ == "__main__":
    main()
