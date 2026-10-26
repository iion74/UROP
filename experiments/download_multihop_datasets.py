"""Download official MuSiQue and 2WikiMultiHopQA benchmark releases.

Examples:
    python -m experiments.download_multihop_datasets --dataset all
    python -m experiments.download_multihop_datasets --dataset musique
    python -m experiments.download_multihop_datasets --dataset 2wiki
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import gdown
import requests


MUSIQUE_DRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
TWOWIKI_URL = "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official multi-hop QA benchmark data")
    parser.add_argument("--dataset", choices=("all", "musique", "2wiki"), default="all")
    parser.add_argument("--output-root", default="data/benchmarks")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    selected = ("musique", "2wiki") if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        target = output_root / ("musique" if dataset == "musique" else "2wikimultihopqa")
        _ensure_missing_target(target)
        with tempfile.TemporaryDirectory(prefix=f"urop_{dataset}_") as temporary_directory:
            archive = Path(temporary_directory) / f"{dataset}.zip"
            if dataset == "musique":
                _download_musique(archive)
                _safe_extract(archive, target, strip_prefix="data")
                source = {
                    "repository": "https://github.com/StonyBrookNLP/musique",
                    "google_drive_file_id": MUSIQUE_DRIVE_ID,
                    "release": "v1.0",
                }
            else:
                _download_2wiki(archive)
                _safe_extract(archive, target)
                source = {
                    "repository": "https://github.com/Alab-NII/2wikimultihop",
                    "download_url": TWOWIKI_URL,
                    "release": "data_ids_april7",
                }
        _write_manifest(target, dataset, source)
        print(f"저장 완료: {target}", flush=True)


def _download_musique(path: Path) -> None:
    print("MuSiQue v1.0 다운로드 시작", flush=True)
    result = gdown.download(id=MUSIQUE_DRIVE_ID, output=str(path), quiet=False)
    if not result or not path.exists():
        raise RuntimeError("MuSiQue 공식 Google Drive 파일을 내려받지 못했습니다.")


def _download_2wiki(path: Path) -> None:
    print("2WikiMultiHopQA data_ids_april7 다운로드 시작", flush=True)
    with requests.get(TWOWIKI_URL, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with path.open("wb") as output:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if not block:
                    continue
                output.write(block)
                downloaded += len(block)
                if total:
                    print(f"  {downloaded / total:.0%} ({downloaded / 1024 / 1024:.0f} MiB)", flush=True)


def _safe_extract(archive: Path, target: Path, strip_prefix: str | None = None) -> None:
    target.mkdir(parents=True, exist_ok=False)
    resolved_target = target.resolve()
    with zipfile.ZipFile(archive) as compressed:
        bad_member = compressed.testzip()
        if bad_member:
            raise ValueError(f"손상된 ZIP 항목입니다: {bad_member}")
        for member in compressed.infolist():
            parts = list(PurePosixPath(member.filename).parts)
            if strip_prefix and parts and parts[0] == strip_prefix:
                parts = parts[1:]
            if not parts or parts[-1] == ".DS_Store":
                continue
            destination = target.joinpath(*parts)
            resolved_destination = destination.resolve()
            if resolved_destination != resolved_target and resolved_target not in resolved_destination.parents:
                raise ValueError(f"안전하지 않은 ZIP 경로입니다: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with compressed.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def _ensure_missing_target(target: Path) -> None:
    if target.exists():
        raise SystemExit(f"대상 폴더가 이미 존재합니다: {target}")


def _write_manifest(target: Path, dataset: str, source: dict[str, str]) -> None:
    files = {
        str(path.relative_to(target)): path.stat().st_size
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    (target / "download_manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "source": source,
                "storage": "official_release_zip",
                "files_bytes": files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
