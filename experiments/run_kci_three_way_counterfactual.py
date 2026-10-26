"""Evaluate a uniform SEARCH/REASON/STOP control at the learned policy's budgets."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("data/benchmarks/multihop/context_reader_aware")
PREFIX = ROOT / "disjoint_holdout_seed20260916_500"
ANALYSIS = ROOT / "analysis" / "disjoint_holdout_seed20260916_500"
MODEL = "qwen3:8b"


def _run(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _main_stage_ready() -> bool:
    path = ANALYSIS / "kci_baselines.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        report.get("split") == "train"
        and all(report.get("datasets", {}).get(name, {}).get("count") == 500 for name in ("hotpotqa", "musique", "2wiki"))
    )


def main() -> None:
    print("세 데이터셋의 고정 정책 평가 완료 대기; 이후 전체 반사실 추론 대조군", flush=True)
    try:
        started = time.monotonic()
        while not _main_stage_ready():
            if time.monotonic() - started > 21600:
                raise TimeoutError("기본 holdout 평가가 6시간 안에 완료되지 않았습니다.")
            time.sleep(30)
        # The first runner unloads Qwen immediately after writing its report.
        time.sleep(60)
        _run(
            "experiments.evaluate_dual_value_reasoning_pilot",
            "--split", "train",
            "--prefix-root", str(PREFIX),
            "--all-questions",
            "--rows-per-dataset", "0",
            "--output-root", str(ANALYSIS / "all_reason"),
            "--ollama-keep-alive", "30m",
            "--log-every", "25",
        )
        _run(
            "experiments.evaluate_dual_value_choice_judge",
            "--split", "train",
            "--prefix-root", str(PREFIX),
            "--pilot-root", str(ANALYSIS / "all_reason"),
            "--output-root", str(ANALYSIS / "all_reason_judge"),
            "--ollama-keep-alive", "30m",
            "--log-every", "25",
        )
        _run(
            "experiments.analyze_kci_required_baselines",
            "--prefix-root", str(PREFIX),
            "--split", "train",
            "--full-predictions", str(ANALYSIS / "full_system.predictions.jsonl"),
            "--all-reason-judge-root", str(ANALYSIS / "all_reason_judge"),
            "--output", str(ANALYSIS / "kci_baselines_matched_three_way.json"),
        )
        print("동일 검색/추론 예산의 무작위 3-way 비교 완료", flush=True)
    finally:
        print("Qwen3:8B GPU 사용 종료", flush=True)
        subprocess.run(["ollama", "stop", MODEL], check=False)


if __name__ == "__main__":
    main()
