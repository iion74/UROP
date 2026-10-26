"""Cheap gold-evidence screen for evidence-conditioned bridge actions.

This experiment does not call the QA reader and does not train a policy.  It
uses train support annotations only to compare candidate retrieval actions,
so expensive answer-level evaluation is reserved for actions that actually
increase support coverage.

Example:
    python -m experiments.evaluate_bridge_retrieval \
      --datasets hotpotqa musique 2wiki \
      --split train \
      --corpus-examples-per-dataset 2000 \
      --max-queries-per-dataset 100
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.evaluate_answer_oracle import _load_fixed_retriever
from experiments.train_multihop_policy import (
    SUPPORTED_DATASETS,
    _corpus_fingerprint,
    _dataset_seed,
    _duration,
    _iter_jsonl,
    evidence_metrics,
    load_processed_examples,
    retrieve_for_action,
    select_query_examples,
)
from src.iterative_retrieval import (
    BridgeSearchConfig,
    compose_bridge_evidence,
    rank_bridge_paths,
)
from src.retriever import GRAPH_FORMAT_VERSION


FORMAT_VERSION = 1
ACTIONS = ("vector", "graph_1hop_large", "bridge_path_1", "bridge_path_2")


def main() -> None:
    args = _arguments()
    started = perf_counter()
    config = BridgeSearchConfig(
        seed_count=args.seed_count,
        graph_hops=args.graph_hops,
        candidates_per_seed=args.candidates_per_seed,
        anchor_text_characters=args.anchor_text_characters,
    )
    config.validate()
    _log("증거 조건부 Bridge 검색 retrieval pilot을 시작합니다. QA reader와 정답 생성은 사용하지 않습니다.")
    _log(
        f"설정: split={args.split}, queries={args.max_queries_per_dataset}, top_k={args.top_k}, "
        f"seed_count={config.seed_count}, graph_hops={config.graph_hops}, "
        f"candidates/seed={config.candidates_per_seed}"
    )

    all_rows: list[dict] = []
    shared_embedding_model = None
    shared_reranker = None
    for dataset in args.datasets:
        input_path = Path(args.processed_root) / dataset / f"{args.split}.jsonl"
        corpus_examples = load_processed_examples(
            input_path,
            args.corpus_examples_per_dataset,
            _dataset_seed(args.corpus_seed, dataset),
        )
        if not corpus_examples:
            raise SystemExit(f"유효한 전처리 문항이 없습니다: {input_path}")
        examples = select_query_examples(
            corpus_examples,
            args.max_queries_per_dataset,
            _dataset_seed(args.query_seed, dataset),
        )
        index_dir = Path(args.index_root) / args.split / dataset
        retriever = _load_fixed_retriever(
            index_dir,
            _corpus_fingerprint(corpus_examples),
            embedding_model=shared_embedding_model,
            reranker=shared_reranker,
        )
        shared_embedding_model = retriever.model or shared_embedding_model
        shared_reranker = retriever.reranker or shared_reranker
        output_path = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
        manifest_config = {
            "format_version": FORMAT_VERSION,
            "experiment": "evidence_conditioned_bridge_retrieval_v1",
            "dataset": dataset,
            "split": args.split,
            "corpus_fingerprint": _corpus_fingerprint(corpus_examples),
            "corpus_seed": args.corpus_seed,
            "query_seed": args.query_seed,
            "corpus_examples_per_dataset": args.corpus_examples_per_dataset,
            "max_queries_per_dataset": args.max_queries_per_dataset,
            "top_k": args.top_k,
            "initial_candidates": args.initial_candidates,
            "actions": list(ACTIONS),
            "graph_format_version": GRAPH_FORMAT_VERSION,
            "bridge_config": asdict(config),
        }
        rows = _evaluate_dataset(
            retriever,
            examples,
            output_path,
            manifest_config,
            config=config,
            top_k=args.top_k,
            initial_candidates=args.initial_candidates,
            log_every=args.log_every,
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)
        _report(rows, dataset)

    _report_macro(all_rows, args.datasets)
    _log(f"Bridge retrieval pilot 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="중간 증거 기반 Bridge 검색의 support-recall 사전 평가")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--output-root", default="data/benchmarks/multihop/bridge_retrieval_pilot")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--max-queries-per-dataset", type=int, default=100)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--initial-candidates", type=int, default=15)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--graph-hops", type=int, default=1)
    parser.add_argument("--candidates-per-seed", type=int, default=12)
    parser.add_argument("--anchor-text-characters", type=int, default=700)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.corpus_examples_per_dataset < 1 or args.max_queries_per_dataset < 1:
        parser.error("corpus와 query 수는 1 이상이어야 합니다.")
    if args.max_queries_per_dataset > args.corpus_examples_per_dataset:
        parser.error("query 수는 corpus 문항 수보다 클 수 없습니다.")
    if args.top_k < 1 or args.initial_candidates < args.top_k or args.log_every < 1:
        parser.error("top-k/log는 1 이상이며 initial candidates는 top-k 이상이어야 합니다.")
    return args


def _evaluate_dataset(
    retriever,
    examples,
    output_path: Path,
    manifest_config: Mapping,
    *,
    config: BridgeSearchConfig,
    top_k: int,
    initial_candidates: int,
    log_every: int,
    overwrite: bool,
) -> list[dict]:
    manifest_path = output_path.with_suffix(".manifest.json")
    cached: dict[str, dict] = {}
    if output_path.exists() or manifest_path.exists():
        if overwrite:
            cached = {}
        elif _manifest_matches(manifest_path, manifest_config):
            cached = {str(row["id"]): row for row in _iter_jsonl(output_path)} if output_path.exists() else {}
        else:
            raise SystemExit(
                f"기존 Bridge retrieval 설정과 현재 설정이 다릅니다: {output_path}\n"
                "다른 --output-root를 쓰거나 --overwrite를 지정하세요."
            )
    expected_ids = {example.policy_id for example in examples}
    rows_by_id = {row_id: row for row_id, row in cached.items() if row_id in expected_ids}
    if len(rows_by_id) == len(examples):
        _log(f"{manifest_config['dataset']}: Bridge retrieval {len(rows_by_id):,}개 전체 재사용")
        return [rows_by_id[example.policy_id] for example in examples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite or not output_path.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    _write_manifest(manifest_path, manifest_config, "in_progress", len(rows_by_id))
    started = perf_counter()
    newly_completed = 0
    with output_path.open(mode, encoding="utf-8") as destination:
        for example in examples:
            if example.policy_id in rows_by_id:
                continue
            initial = retriever.search(example.question, top_k=initial_candidates)
            profile = retriever._evidence_profile(initial)
            graph_sources, graph_extra = retrieve_for_action(
                retriever,
                example.question,
                initial,
                profile,
                "graph_1hop_large",
                top_k,
            )
            bridge = rank_bridge_paths(retriever, example.question, initial, config)
            sources = {
                "vector": initial[:top_k],
                "graph_1hop_large": graph_sources,
                "bridge_path_1": compose_bridge_evidence(initial, bridge.paths, top_k=top_k, max_paths=1),
                "bridge_path_2": compose_bridge_evidence(initial, bridge.paths, top_k=top_k, max_paths=2),
            }
            actions = {
                action: _score_retrieval(
                    action_sources,
                    example.supporting_groups,
                    extra_candidates=(graph_extra if action == "graph_1hop_large" else bridge.distinct_candidates),
                )
                for action, action_sources in sources.items()
            }
            oracle_action = max(
                ACTIONS,
                key=lambda action: (
                    float(actions[action]["all_support_recall"]),
                    float(actions[action]["source_recall"]),
                    -ACTIONS.index(action),
                ),
            )
            row = {
                "format_version": FORMAT_VERSION,
                "id": example.policy_id,
                "dataset": example.dataset,
                "question": example.question,
                "hop_count": example.hop_count,
                "reasoning_type": example.reasoning_type,
                "actions": actions,
                "oracle_action": oracle_action,
                "bridge_audit": {
                    "evaluated_pairs": bridge.evaluated_pairs,
                    "distinct_candidates": bridge.distinct_candidates,
                    "top_paths": [
                        {
                            "anchor_content_id": path.anchor.source,
                            "candidate_content_id": path.candidate.source,
                            "score": path.score,
                            "conditioned_score": path.conditioned_score,
                            "title_mentioned": path.title_mentioned,
                            "distance": path.distance,
                        }
                        for path in bridge.paths[:4]
                    ],
                },
            }
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows_by_id[example.policy_id] = row
            newly_completed += 1
            completed = len(rows_by_id)
            if completed % log_every == 0 or completed == len(examples):
                elapsed = perf_counter() - started
                eta = elapsed / max(1, newly_completed) * (len(examples) - completed)
                current = list(rows_by_id.values())
                _log(
                    f"{example.dataset} retrieval {completed:,}/{len(examples):,} "
                    f"({completed / len(examples):.1%}) | ETA {_duration(eta)} | "
                    f"All-Support Direct/Graph/Bridge1/Bridge2="
                    f"{_mean(current, 'vector', 'all_support_recall'):.3f}/"
                    f"{_mean(current, 'graph_1hop_large', 'all_support_recall'):.3f}/"
                    f"{_mean(current, 'bridge_path_1', 'all_support_recall'):.3f}/"
                    f"{_mean(current, 'bridge_path_2', 'all_support_recall'):.3f}"
                )
                _write_manifest(manifest_path, manifest_config, "in_progress", completed)
    _write_manifest(manifest_path, manifest_config, "ready", len(rows_by_id))
    return [rows_by_id[example.policy_id] for example in examples]


def _score_retrieval(sources, supporting_groups, *, extra_candidates: int) -> dict:
    retrieved = {chunk.source for chunk, _ in sources}
    source_recall, all_support = evidence_metrics(retrieved, supporting_groups)
    return {
        "source_recall": source_recall,
        "all_support_recall": all_support,
        "extra_candidates": int(extra_candidates),
        "evidence_content_ids": [chunk.source for chunk, _ in sources],
    }


def _report(rows: list[dict], name: str) -> None:
    distribution = dict(sorted(Counter(str(row["oracle_action"]) for row in rows).items()))
    parts = [
        f"{action} Recall/All={_mean(rows, action, 'source_recall'):.3f}/"
        f"{_mean(rows, action, 'all_support_recall'):.3f}"
        for action in ACTIONS
    ]
    _log(f"{name}: n={len(rows):,} | " + " | ".join(parts) + f" | Retrieval Oracle={distribution}")


def _report_macro(rows: list[dict], datasets: Iterable[str]) -> None:
    groups = [[row for row in rows if row["dataset"] == dataset] for dataset in datasets]
    groups = [group for group in groups if group]
    if not groups:
        return
    parts = []
    for action in ACTIONS:
        recall = float(np.mean([_mean(group, action, "source_recall") for group in groups]))
        all_support = float(np.mean([_mean(group, action, "all_support_recall") for group in groups]))
        parts.append(f"{action} Recall/All={recall:.3f}/{all_support:.3f}")
    _log("Macro: " + " | ".join(parts))


def _mean(rows: list[dict], action: str, metric: str) -> float:
    return float(np.mean([float(row["actions"][action][metric]) for row in rows])) if rows else 0.0


def _manifest_matches(path: Path, expected: Mapping) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        # Hash-ranked query cohorts are nested for a fixed query seed.  Allow a
        # 20-row smoke run to grow to 100/1,000 rows without recomputing it.
        ignored = {"max_queries_per_dataset", "status", "completed_rows"}
        return all(saved.get(key) == value for key, value in expected.items() if key not in ignored)
    except (OSError, ValueError, TypeError):
        return False


def _write_manifest(path: Path, config: Mapping, status: str, completed: int) -> None:
    payload = dict(config)
    payload.update({"status": status, "completed_rows": completed})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
