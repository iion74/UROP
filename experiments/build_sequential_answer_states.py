"""Build fair sequential-policy states from cached one-shot Answer Oracle rows.

The one-shot audit already contains the frozen reader result and provenance for
all five terminal retrieval actions.  Sequential routing only needs to add the
post-1-hop evidence-state features and state/action labels.  Reusing the cached
terminal evidence makes the one-shot and sequential Oracle exactly comparable
and avoids any additional reader generation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.evaluate_answer_oracle import (
    DIRECT_ACTION,
    FIRST_STEP_ACTIONS,
    SEQUENTIAL_OUTPUT_FORMAT_VERSION,
    SEQUENTIAL_STATE_FEATURE_NAMES,
    _evaluate_sequential_trajectories,
)
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from src.policy import POLICY_FEATURE_NAMES
from src.retriever import SemanticRetriever
from src.routing import POLICY_ACTION_IDS, EvidenceProfile


def main() -> None:
    args = _arguments()
    started = perf_counter()
    print(
        "공유 terminal evidence 기반 Sequential 상태 생성을 시작합니다. "
        "추가 QA reader 호출은 없습니다.",
        flush=True,
    )
    total_rows: list[dict] = []
    for dataset in args.datasets:
        input_path = Path(args.answer_root) / f"{dataset}_{args.split}.jsonl"
        input_manifest_path = input_path.with_suffix(".manifest.json")
        rows = list(_iter_jsonl(input_path))
        if not rows:
            raise SystemExit(f"one-shot Answer Oracle 결과가 없습니다: {input_path}")
        if args.max_queries_per_dataset:
            rows = rows[: args.max_queries_per_dataset]

        manifest = _read_manifest(input_manifest_path)
        _validate_one_shot_rows(rows, dataset)
        index_dir = Path(args.index_root) / args.split / dataset
        retriever = SemanticRetriever(index_dir)
        if not retriever.load():
            raise SystemExit(f"고정 corpus 인덱스를 읽지 못했습니다: {index_dir}")
        chunk_by_id = {chunk.chunk_id: chunk for chunk in retriever.chunks}

        converted: list[dict] = []
        for position, row in enumerate(rows, start=1):
            converted.append(_convert_row(retriever, chunk_by_id, row))
            if position % args.log_every == 0 or position == len(rows):
                print(
                    f"{dataset} 상태 {position:,}/{len(rows):,} ({position / len(rows):.1%})",
                    flush=True,
                )

        output_path = Path(args.output_root) / f"{dataset}_{args.split}_sequential_v6.jsonl"
        _write_rows(output_path, converted, overwrite=args.overwrite)
        _write_manifest(
            output_path.with_suffix(".manifest.json"),
            manifest,
            input_path=input_path,
            row_count=len(converted),
        )
        _report(converted, dataset)
        total_rows.extend(converted)

    _report(total_rows, "전체 Micro")
    print(f"Sequential 상태 생성 완료 · 총 소요 {_duration(perf_counter() - started)}", flush=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="one-shot cache에서 공정한 순차 정책 상태 생성")
    parser.add_argument("--answer-root", default="data/benchmarks/multihop/answer_oracle")
    parser.add_argument("--output-root", default="data/benchmarks/multihop/answer_oracle")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--max-queries-per-dataset", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_queries_per_dataset < 0 or args.log_every < 1:
        parser.error("query 수는 0 이상, log 간격은 1 이상이어야 합니다.")
    return args


def _convert_row(
    retriever: SemanticRetriever,
    chunk_by_id: dict,
    row: dict,
) -> dict:
    initial_profile = _profile_from_row(row)
    sources = {
        action: _sources_from_evidence(chunk_by_id, row["actions"][action]["evidence"])
        for action in (DIRECT_ACTION, *FIRST_STEP_ACTIONS)
    }
    sequential = _evaluate_sequential_trajectories(
        retriever,
        initial_profile=initial_profile,
        initial_sources=sources[DIRECT_ACTION],
        first_sources={action: sources[action] for action in FIRST_STEP_ACTIONS},
        action_results=row["actions"],
    )
    converted = dict(row)
    converted["format_version"] = SEQUENTIAL_OUTPUT_FORMAT_VERSION
    converted["sequential"] = sequential
    converted["sequential_oracle_trajectory"] = sequential["oracle_trajectory"]
    converted["sequential_oracle_em"] = sequential["oracle_em"]
    converted["sequential_oracle_f1"] = sequential["oracle_f1"]
    converted["sequential_oracle_em_gain"] = sequential["oracle_em"] - row["direct_em"]
    converted["sequential_oracle_f1_gain"] = sequential["oracle_f1"] - row["direct_f1"]
    return converted


def _profile_from_row(row: dict) -> EvidenceProfile:
    features = row["features"]
    missing = [name for name in POLICY_FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"{row.get('id')}: 초기 evidence feature가 없습니다: {missing}")
    return EvidenceProfile(
        score_concentration=float(features["score_concentration"]),
        score_margin=float(features["score_margin"]),
        source_dispersion=float(features["source_dispersion"]),
        graph_connectivity=float(features["graph_connectivity"]),
        redundancy=float(features["redundancy"]),
        one_hop_gain=float(features["one_hop_gain"]),
        two_hop_gain=float(features["two_hop_gain"]),
    )


def _sources_from_evidence(chunk_by_id: dict, evidence: list[dict]) -> list[tuple]:
    sources = []
    for item in evidence:
        chunk_id = str(item["chunk_id"])
        try:
            chunk = chunk_by_id[chunk_id]
        except KeyError as error:
            raise ValueError(f"고정 인덱스에 evidence chunk가 없습니다: {chunk_id}") from error
        sources.append((chunk, float(item["score"])))
    return sources


def _validate_one_shot_rows(rows: list[dict], dataset: str) -> None:
    ids: set[str] = set()
    for row in rows:
        if row.get("dataset") != dataset:
            raise ValueError(f"{dataset} 파일에 다른 데이터셋 행이 있습니다: {row.get('id')}")
        if row.get("id") in ids:
            raise ValueError(f"중복 질문 ID입니다: {row.get('id')}")
        ids.add(str(row.get("id")))
        missing = [action for action in POLICY_ACTION_IDS if action not in row.get("actions", {})]
        if missing:
            raise ValueError(f"{row.get('id')}: terminal 행동이 없습니다: {missing}")


def _read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"one-shot manifest를 읽지 못했습니다: {path}") from error
    if int(manifest.get("format_version", -1)) != 2:
        raise SystemExit(f"지원하지 않는 one-shot 결과 형식입니다: {path}")
    return manifest


def _write_rows(path: Path, rows: list[dict], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"출력이 이미 있습니다: {path}\n다시 만들려면 --overwrite를 지정하세요.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _write_manifest(
    path: Path,
    one_shot_manifest: dict,
    *,
    input_path: Path,
    row_count: int,
) -> None:
    payload = {
        key: value
        for key, value in one_shot_manifest.items()
        if key not in {"status", "cached_rows", "last_requested_rows", "last_requested_completed"}
    }
    payload.update(
        {
            "format_version": SEQUENTIAL_OUTPUT_FORMAT_VERSION,
            "sequential_trajectories": True,
            "construction": "shared_terminal_state_v1",
            "sequential_state_feature_names": list(SEQUENTIAL_STATE_FEATURE_NAMES),
            "source_answer_oracle_file": str(input_path),
            "status": "ready",
            "cached_rows": row_count,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _report(rows: list[dict], name: str) -> None:
    if not rows:
        return
    direct = float(np.mean([row["direct_f1"] for row in rows]))
    one_shot = float(np.mean([row["oracle_f1"] for row in rows]))
    sequential = float(np.mean([row["sequential_oracle_f1"] for row in rows]))
    paths = Counter(row["sequential_oracle_trajectory"] for row in rows)
    matched = all(np.isclose(row["oracle_f1"], row["sequential_oracle_f1"]) for row in rows)
    print(
        f"{name}: n={len(rows):,} | Direct/One-shot/공유 Terminal Oracle F1="
        f"{direct:.3f}/{one_shot:.3f}/{sequential:.3f} | Oracle 일치={'yes' if matched else 'NO'} "
        f"| trajectory={dict(sorted(paths.items()))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
