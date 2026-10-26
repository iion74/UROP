"""Package existing research artifacts; never train a model or call an LLM."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RELEASE = ROOT / "artifacts/release"
SKIP_DIRS = {".git", ".venv", ".aws", ".codex", ".agents", "__pycache__", ".pytest_cache", "artifacts"}
MODEL_SUFFIXES = {".joblib", ".npz", ".pt", ".pth", ".safetensors"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_files() -> list[Path]:
    paths = []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS)
        paths.extend(Path(base) / name for name in sorted(names) if not (Path(base) / name).is_symlink())
    return paths


def export_catalog() -> None:
    lines = ["# 실험 코드 안내", "", "## 최종 논문 경로", "",
             "최종 방법은 `sequential_best_future_f1_gain_hgb`다. 아래 여섯 개가 주 경로다.", "",
             "| 파일 | 역할 |", "|---|---|",
             "| `evaluate_context_reader_aware_prefixes.py` | 질문별 공식 후보 문서에서 단계별 기록 생성 |",
             "| `evaluate_sequential_answer_gain_policy.py` | 105차원 이득 회귀 정책 학습·평가 |",
             "| `run_sequential_answer_gain_multiseed.py` | 5개 표본 시드 순차 실행 |",
             "| `analyze_sequential_answer_gain_multiseed.py` | 시드·문항 수 감사와 집계 |",
             "| `export_paper_seed_confidence_intervals.py` | 기존 결과의 신뢰구간 증빙 추출 |",
             "| `scripts/replay_published_policy.py` | 공개 수치형 입력으로 정책 점수 재생 |", "",
             "## 전체 실험 모듈", "",
             "원래 파일과 이력을 보존한다. 아래 목록의 이전 실험은 최종 비교군이나 채택된 방법을 뜻하지 않는다.", "",
             "| 모듈 | 파일에 기록된 설명 |", "|---|---|"]
    for path in sorted((ROOT / "experiments").glob("*.py")):
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or "설명 없음"
        first = doc.splitlines()[0].replace("|", "\\|")
        lines.append(f"| [{path.name}](../experiments/{path.name}) | {first} |")
    (ROOT / "docs/EXPERIMENT_CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_versions() -> None:
    names = ["streamlit", "sentence-transformers", "faiss-cpu", "numpy", "PyMuPDF", "pandas",
             "requests", "python-dotenv", "rank-bm25", "scikit-learn", "joblib", "matplotlib",
             "datasets", "gdown", "torch", "pytest"]
    lines = ["# Installed package snapshot at publication export; not the original experiment lockfile.",
             f"# Python {sys.version.split()[0]}"]
    for name in names:
        try:
            lines.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"# {name}: not installed")
    (ROOT / "requirements-research-snapshot.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_files() -> list[Path]:
    return sorted(path for path in (ROOT / "data/models").rglob("*")
                  if path.is_file() and path.suffix in MODEL_SUFFIXES)


def export_inventory() -> None:
    paths = source_files()
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    result = subprocess.run(["git", "check-ignore", "--no-index", "-z", "--stdin"],
                            cwd=ROOT, input=("\0".join(relative) + "\0").encode(),
                            capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode())
    ignored = set(result.stdout.decode().strip("\0").split("\0"))
    rows = []
    for path, name in zip(paths, relative):
        if name not in ignored:
            location = "git_repository"
        elif name.startswith("data/models/") and path.suffix in MODEL_SUFFIXES:
            location = "release_policy_models"
        elif path.name.startswith(".env"):
            location = "excluded_credentials"
        elif "study_progress" in path.name:
            location = "excluded_private_study_log"
        elif name.startswith("data/cache/") or "/index/" in name or path.suffix in {".sqlite3", ".faiss"}:
            location = "excluded_regenerable_cache_or_index"
        elif name.startswith("data/01."):
            location = "external_ncs_source_not_redistributed"
        elif name.startswith(("data/benchmarks/processed", "data/benchmarks/hotpotqa/",
                              "data/benchmarks/musique/", "data/benchmarks/2wikimultihopqa/")):
            location = "official_download_or_preprocessing"
        else:
            location = "local_intermediate_document_records"
        # The inventory cannot contain its own stable hash.
        hashed = digest(path) if location in {"git_repository", "release_policy_models"} and name != "docs/workspace_inventory.csv" else ""
        rows.append({"path": name, "bytes": path.stat().st_size, "publication": location, "sha256": hashed})
    output = ROOT / "docs/workspace_inventory.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["path", "bytes", "publication", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Inventory: {len(rows)} files", flush=True)


def export_numeric() -> None:
    import numpy as np
    from experiments.evaluate_sequential_answer_gain_policy import _feature_names, _read

    base = ROOT / "data/benchmarks/multihop/context_reader_aware"
    schema = tuple(json.loads((base / "train_pool_seed42_2000/hotpotqa_train.manifest.json").read_text())["feature_names"])
    tasks = [("full_validation", "validation")] + [(f"train_pool_seed{seed}_2000", "train") for seed in range(42, 47)]
    for cohort, split in tasks:
        datasets = _read(base / cohort, split, schema)
        destination = ROOT / "artifacts/replay" / cohort
        destination.mkdir(parents=True, exist_ok=True)
        for dataset, data in datasets.items():
            arrays = {key: np.asarray(value) for key, value in data.items()}
            arrays["feature_names"] = np.asarray(_feature_names(schema))
            if any(array.dtype.hasobject for array in arrays.values()):
                raise ValueError("Object arrays must not be published")
            np.savez_compressed(destination / f"{dataset}.npz", **arrays)
        print(f"Numeric export complete: {cohort}", flush=True)


def archive(name: str, paths: list[Path]) -> dict:
    destination = RELEASE / name
    if destination.exists():
        raise FileExistsError(f"Preserving existing bundle: {destination}")
    with tarfile.open(destination, "w:gz", compresslevel=1) as bundle:
        for path in paths:
            bundle.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    record = {"name": name, "bytes": destination.stat().st_size, "sha256": digest(destination),
              "file_count": len(paths)}
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", action="store_true", help="Also export numeric states and model archives")
    args = parser.parse_args()
    export_catalog()
    export_versions()
    export_inventory()
    if args.bundles:
        RELEASE.mkdir(parents=True, exist_ok=True)
        export_numeric()
        files = model_files()
        records = [archive("urop-policy-models.tar.gz", files),
                   archive("urop-numeric-replay.tar.gz", sorted((ROOT / "artifacts/replay").rglob("*.npz")))]
        manifest = {"snapshot_date": "2026-10-26", "new_training": False, "llm_calls": 0,
                    "includes_source_document_text": False, "assets": records}
        (RELEASE / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (RELEASE / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['name']}\n" for r in records))


if __name__ == "__main__":
    main()
