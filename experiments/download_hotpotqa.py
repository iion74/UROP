"""Download and persist an official HotpotQA configuration for local experiments.

Example:
    python -m experiments.download_hotpotqa \
      --output-dir data/benchmarks/hotpotqa/distractor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HotpotQA into a local benchmark directory")
    parser.add_argument("--config", default="distractor", choices=("distractor", "fullwiki"))
    parser.add_argument("--output-dir", default="data/benchmarks/hotpotqa/distractor")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"대상 폴더가 이미 비어 있지 않습니다: {output_dir}")

    dataset = load_dataset("hotpotqa/hotpot_qa", args.config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    manifest = {
        "dataset": "hotpotqa/hotpot_qa",
        "config": args.config,
        "splits": {name: len(split) for name, split in dataset.items()},
        "storage": "datasets.save_to_disk",
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"저장 완료: {output_dir}")
    print(f"split별 행 수: {manifest['splits']}")


if __name__ == "__main__":
    main()
