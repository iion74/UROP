"""Create answer-level action labels in each benchmark's official context.

Unlike the early fixed-corpus pilot, this command never pools paragraphs from
different questions.  Every query is evaluated against only the distractor
context shipped with that benchmark example (HotpotQA/2Wiki: about 10
paragraphs, MuSiQue: about 20).  This is the scalable, benchmark-faithful path
for final policy training and unseen-validation evaluation.

The command is intentionally resumable.  Each completed row is flushed to a
compact JSONL file and a rerun with the same manifest skips it.  The frozen QA
reader is the expensive part; no model is trained by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.evaluate_answer_oracle import (
    DIRECT_ACTION,
    OUTPUT_FORMAT_VERSION,
    _evaluate_example,
)
from experiments.train_multihop_policy import (
    QUERY_SAMPLING_VERSION,
    SUPPORTED_DATASETS,
    MultiHopExample,
    _dataset_seed,
    _duration,
    _iter_jsonl,
    _normalise_processed_record,
    load_processed_examples,
)
from src.document_loader import SourceDocument, chunk_documents
from src.policy import POLICY_FEATURE_NAMES
from src.qa_evaluation import (
    PROMPT_VERSION,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    ollama_model_identity,
)
from src.retriever import GRAPH_FORMAT_VERSION, SemanticRetriever
from src.routing import POLICY_ACTION_IDS
from src.settings import settings


CONTEXT_MODE = "official_per_question_distractor_v1"
COMPACT_FORMAT_VERSION = 1


def main() -> None:
    args = _arguments()
    started = perf_counter()
    _log("공식 문항별 context Answer Oracle 생성을 시작합니다. 검색 행동 외 모델은 모두 동결됩니다.")

    session = requests.Session()
    try:
        resolved_model, model_digest = ollama_model_identity(
            args.ollama_host,
            args.reader_model,
            session=session,
        )
    except RuntimeError as error:
        raise SystemExit(
            f"{error}\nOllama가 이미 실행 중인지 'ollama list'로 확인하세요."
        ) from error
    reader_config = ReaderConfig(
        host=args.ollama_host,
        model=resolved_model,
        model_digest=model_digest,
        max_tokens=args.reader_max_tokens,
        timeout_seconds=args.reader_timeout,
        seed=args.reader_seed,
        temperature=0.0,
        keep_alive=args.ollama_keep_alive,
    )
    _log(
        f"Reader 고정: model={resolved_model}, digest={model_digest or '미제공'}, "
        f"temperature=0, prompt={PROMPT_VERSION}"
    )
    _log(
        f"검색 고정: context={CONTEXT_MODE}, top_k={args.top_k}, "
        f"initial_candidates={args.initial_candidates}, graph_semantic_neighbors="
        f"{args.graph_semantic_neighbors}"
    )

    shared_embedding_model = None
    shared_reranker = None
    all_rows: list[dict] = []
    with AnswerCache(Path(args.cache_path)) as cache:
        reader = FrozenOllamaReader(reader_config, cache, session=session)
        for dataset in args.datasets:
            input_path = Path(args.processed_root) / dataset / f"{args.split}.jsonl"
            if not input_path.exists():
                raise SystemExit(f"전처리 데이터가 없습니다: {input_path}")
            examples, total = _example_stream(
                input_path,
                maximum=args.max_queries_per_dataset,
                seed=_dataset_seed(args.query_seed, dataset),
            )
            config = _output_config(
                args,
                dataset=dataset,
                input_path=input_path,
                total=total,
                reader_config=reader_config,
            )
            output_path = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
            rows, shared_embedding_model, shared_reranker = _evaluate_dataset(
                examples,
                total,
                output_path,
                config,
                reader=reader,
                embedding_model=shared_embedding_model,
                reranker=shared_reranker,
                args=args,
            )
            all_rows.extend(rows)
            _report(rows, dataset)

    _report_macro(all_rows, args.datasets)
    _log(f"공식 context Answer Oracle 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HotpotQA·MuSiQue·2Wiki 공식 distractor context Answer Oracle"
    )
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument(
        "--output-root",
        default="data/benchmarks/multihop/context_answer_oracle",
    )
    parser.add_argument(
        "--cache-path",
        default="data/benchmarks/multihop/context_answer_cache.sqlite3",
    )
    parser.add_argument(
        "--max-queries-per-dataset",
        type=int,
        default=0,
        help="0이면 해당 split의 모든 문항, 양수면 재현 가능한 pilot sample",
    )
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--initial-candidates",
        type=int,
        default=5,
        help="Direct frontier 크기. Graph는 이 frontier 밖의 문맥만 확장합니다.",
    )
    parser.add_argument("--graph-semantic-neighbors", type=int, default=3)
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--actions", nargs="+", choices=POLICY_ACTION_IDS, default=list(POLICY_ACTION_IDS))
    parser.add_argument(
        "--include-gold-support",
        action="store_true",
        help="gold-support reader ceiling도 생성(최종 validation에 권장, train은 생략 권장)",
    )
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--reader-model", default=settings.ollama_model)
    parser.add_argument("--reader-max-tokens", type=int, default=64)
    parser.add_argument("--reader-timeout", type=int, default=300)
    parser.add_argument("--reader-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    args.actions = tuple(dict.fromkeys(args.actions))
    if DIRECT_ACTION not in args.actions:
        parser.error("Direct baseline인 vector 행동이 필요합니다.")
    if args.max_queries_per_dataset < 0:
        parser.error("max queries는 0 이상이어야 합니다.")
    if args.top_k < 1 or args.initial_candidates < args.top_k:
        parser.error("top-k는 1 이상이고 initial-candidates는 top-k 이상이어야 합니다.")
    if args.graph_semantic_neighbors < 1 or args.log_every < 1:
        parser.error("graph neighbors와 log 간격은 1 이상이어야 합니다.")
    if args.context_character_budget < 1 or args.reader_max_tokens < 1 or args.reader_timeout < 1:
        parser.error("context/token/timeout 설정은 1 이상이어야 합니다.")
    return args


def _example_stream(path: Path, *, maximum: int, seed: int) -> tuple[Iterable[MultiHopExample], int]:
    if maximum:
        examples = load_processed_examples(path, maximum, seed)
        return examples, len(examples)
    total = _manifest_row_count(path)
    return _iter_processed_examples(path), total


def _iter_processed_examples(path: Path) -> Iterator[MultiHopExample]:
    for record in _iter_jsonl(path):
        example = _normalise_processed_record(record)
        if example is not None:
            yield example


def _manifest_row_count(path: Path) -> int:
    manifest_path = path.with_suffix(".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = int(manifest["stats"]["rows"])
        if rows > 0:
            return rows
    except (OSError, ValueError, TypeError, KeyError):
        pass
    with path.open("rb") as source:
        return sum(1 for line in source if line.strip())


def _evaluate_dataset(
    examples: Iterable[MultiHopExample],
    total: int,
    output_path: Path,
    config: Mapping,
    *,
    reader: FrozenOllamaReader,
    embedding_model,
    reranker,
    args: argparse.Namespace,
) -> tuple[list[dict], object, object]:
    manifest_path = output_path.with_suffix(".manifest.json")
    cached_rows: list[dict] = []
    if output_path.exists() or manifest_path.exists():
        if args.overwrite:
            cached_rows = []
        elif _matching_manifest(manifest_path, config):
            cached_rows = list(_iter_jsonl(output_path)) if output_path.exists() else []
        else:
            raise SystemExit(
                f"기존 공식-context Oracle 설정이 현재 실행과 다릅니다: {output_path}\n"
                "다른 --output-root를 쓰거나 --overwrite를 지정하세요."
            )
    completed_ids = {str(row["id"]) for row in cached_rows}
    if len(completed_ids) != len(cached_rows):
        raise SystemExit(f"중복 ID가 있는 Oracle 결과입니다: {output_path}")
    if len(cached_rows) > total:
        raise SystemExit(f"캐시 행이 요청 문항보다 많습니다: {output_path}")
    if len(cached_rows) == total:
        _log(f"{config['dataset']}: {total:,}개 Oracle 행 전체 재사용")
        return cached_rows, embedding_model, reranker

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output_path.exists() else "a"
    if mode == "w":
        cached_rows = []
        completed_ids = set()
    _write_manifest(manifest_path, config, status="in_progress", completed=len(cached_rows), total=total)

    rows = list(cached_rows)
    stats = _RunningStats.from_rows(rows, actions=args.actions)
    started = perf_counter()
    newly_completed = 0
    last_retriever: SemanticRetriever | None = None
    with output_path.open(mode, encoding="utf-8") as destination:
        for example in examples:
            if example.policy_id in completed_ids:
                continue
            retriever = _local_retriever(
                example,
                embedding_model=embedding_model,
                reranker=reranker,
                semantic_neighbors=args.graph_semantic_neighbors,
            )
            row = _evaluate_example(
                retriever,
                reader,
                example,
                actions=args.actions,
                top_k=args.top_k,
                context_character_budget=args.context_character_budget,
                initial_candidates=args.initial_candidates,
                include_gold_support=args.include_gold_support,
            )
            row = _compact_row(row)
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows.append(row)
            completed_ids.add(example.policy_id)
            stats.add(row)
            newly_completed += 1
            embedding_model = retriever.model or embedding_model
            reranker = retriever.reranker or reranker
            last_retriever = retriever

            completed = len(rows)
            if completed % args.log_every == 0 or completed == total:
                elapsed = perf_counter() - started
                eta = elapsed / max(1, newly_completed) * max(0, total - completed)
                _log(
                    f"{example.dataset} Answer {completed:,}/{total:,} ({completed / total:.1%}) "
                    f"| ETA {_duration(eta)} | Direct F1={stats.direct_f1:.3f} "
                    f"| Oracle F1={stats.oracle_f1:.3f} | Graph-beneficial="
                    f"{stats.graph_beneficial:.1%} | reader cache hit={stats.cache_rate:.0%}"
                )
                _write_manifest(
                    manifest_path,
                    config,
                    status="in_progress",
                    completed=completed,
                    total=total,
                )

    if len(rows) != total:
        raise SystemExit(
            f"{config['dataset']}: 전처리 manifest는 {total:,}개인데 유효 Oracle 행은 "
            f"{len(rows):,}개입니다. 전처리 데이터를 점검하세요."
        )
    _write_manifest(manifest_path, config, status="ready", completed=len(rows), total=total)
    if last_retriever is not None:
        embedding_model = last_retriever.model or embedding_model
        reranker = last_retriever.reranker or reranker
    return rows, embedding_model, reranker


def _local_retriever(
    example: MultiHopExample,
    *,
    embedding_model,
    reranker,
    semantic_neighbors: int,
) -> SemanticRetriever:
    documents = [
        SourceDocument(source=document.content_id, title=document.title, text=document.text)
        for document in example.documents
    ]
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    retriever = SemanticRetriever(Path(".context_answer_oracle_in_memory"))
    retriever.model = embedding_model
    retriever.reranker = reranker
    retriever.build(chunks, persist=False, semantic_neighbors=semantic_neighbors)
    return retriever


def _compact_row(row: dict) -> dict:
    """Remove bulky raw generations while retaining audit and policy fields."""

    keep_result = (
        "prediction",
        "matched_gold",
        "em",
        "f1",
        "source_recall",
        "all_support_recall",
        "extra_candidates",
        "initial_retrieval_latency_seconds",
        "action_retrieval_latency_seconds",
        "reader_latency_seconds",
        "reader_cache_hit",
        "context_hash",
        "context_characters",
    )
    compact = dict(row)
    compact["compact_format_version"] = COMPACT_FORMAT_VERSION
    compact["actions"] = {
        action: {
            **{key: result[key] for key in keep_result if key in result},
            "evidence_content_ids": [str(item["content_id"]) for item in result.get("evidence", ())],
        }
        for action, result in row["actions"].items()
    }
    if "gold_support" in row:
        result = row["gold_support"]
        compact["gold_support"] = {
            **{key: result[key] for key in keep_result if key in result},
            "evidence_content_ids": [str(item["content_id"]) for item in result.get("evidence", ())],
        }
    return compact


class _RunningStats:
    def __init__(self, actions: Iterable[str]) -> None:
        self.actions = tuple(actions)
        self.rows = 0
        self.direct_f1_total = 0.0
        self.oracle_f1_total = 0.0
        self.graph_beneficial_total = 0
        self.cache_hits = 0
        self.reader_calls = 0

    @classmethod
    def from_rows(cls, rows: Iterable[dict], *, actions: Iterable[str]) -> "_RunningStats":
        stats = cls(actions)
        for row in rows:
            stats.add(row)
        return stats

    def add(self, row: Mapping) -> None:
        self.rows += 1
        self.direct_f1_total += float(row["direct_f1"])
        self.oracle_f1_total += float(row["oracle_f1"])
        self.graph_beneficial_total += int(float(row["oracle_f1_gain"]) > 1e-9)
        results = list(row["actions"].values())
        if row.get("gold_support"):
            results.append(row["gold_support"])
        self.cache_hits += sum(int(result.get("reader_cache_hit", False)) for result in results)
        self.reader_calls += len(results)

    @property
    def direct_f1(self) -> float:
        return self.direct_f1_total / max(1, self.rows)

    @property
    def oracle_f1(self) -> float:
        return self.oracle_f1_total / max(1, self.rows)

    @property
    def graph_beneficial(self) -> float:
        return self.graph_beneficial_total / max(1, self.rows)

    @property
    def cache_rate(self) -> float:
        return self.cache_hits / max(1, self.reader_calls)


def _output_config(
    args: argparse.Namespace,
    *,
    dataset: str,
    input_path: Path,
    total: int,
    reader_config: ReaderConfig,
) -> dict:
    return {
        "format_version": OUTPUT_FORMAT_VERSION,
        "compact_format_version": COMPACT_FORMAT_VERSION,
        "dataset": dataset,
        "split": args.split,
        "status": "configured",
        "context_mode": CONTEXT_MODE,
        "processed_input": str(input_path),
        "processed_fingerprint": _file_fingerprint(input_path),
        "query_seed": args.query_seed,
        "query_sampling_version": QUERY_SAMPLING_VERSION,
        "max_queries_per_dataset": args.max_queries_per_dataset,
        "requested_rows": total,
        "top_k": args.top_k,
        "initial_candidates": args.initial_candidates,
        "context_character_budget": args.context_character_budget,
        "actions": list(args.actions),
        "include_gold_support": args.include_gold_support,
        "feature_names": list(POLICY_FEATURE_NAMES),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "reranker_candidates": settings.reranker_candidates,
        "graph_format_version": GRAPH_FORMAT_VERSION,
        "graph_semantic_neighbors": args.graph_semantic_neighbors,
        "graph_title_bridges": settings.graph_title_bridges,
        "graph_max_title_bridges_per_chunk": settings.graph_max_title_bridges_per_chunk,
        "router_seed_count": settings.router_seed_count,
        "graph_small_budget": settings.graph_small_budget,
        "graph_large_budget": settings.graph_large_budget,
        "graph_expansion_decay": settings.graph_expansion_decay,
        "graph_rerank_prior_weight": settings.graph_rerank_prior_weight,
        "graph_source_diversity_bonus": settings.graph_source_diversity_bonus,
        "graph_min_expanded_results": settings.graph_min_expanded_results,
        "reader": {**asdict(reader_config), "prompt_version": PROMPT_VERSION},
    }


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matching_manifest(path: Path, expected: Mapping) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        ignored = {"status", "completed_rows", "updated_at"}
        return all(saved.get(key) == value for key, value in expected.items() if key not in ignored)
    except (OSError, ValueError, TypeError):
        return False


def _write_manifest(
    path: Path,
    config: Mapping,
    *,
    status: str,
    completed: int,
    total: int,
) -> None:
    payload = dict(config)
    payload.update({"status": status, "completed_rows": completed, "requested_rows": total})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _report(rows: list[dict], name: str) -> dict:
    direct_em = float(np.mean([row["direct_em"] for row in rows]))
    direct_f1 = float(np.mean([row["direct_f1"] for row in rows]))
    oracle_em = float(np.mean([row["oracle_em"] for row in rows]))
    oracle_f1 = float(np.mean([row["oracle_f1"] for row in rows]))
    graph_rate = float(np.mean([row["oracle_action"] != DIRECT_ACTION for row in rows]))
    distribution = dict(sorted(Counter(row["oracle_action"] for row in rows).items()))
    gold = ""
    if all("gold_support_f1" in row for row in rows):
        gold = (
            f" | Gold-support EM/F1={float(np.mean([row['gold_support_em'] for row in rows])):.3f}/"
            f"{float(np.mean([row['gold_support_f1'] for row in rows])):.3f}"
        )
    _log(
        f"{name}: n={len(rows):,} | Direct EM/F1={direct_em:.3f}/{direct_f1:.3f} "
        f"| Oracle EM/F1={oracle_em:.3f}/{oracle_f1:.3f} "
        f"| gain={oracle_em - direct_em:+.3f}/{oracle_f1 - direct_f1:+.3f} "
        f"| Graph-beneficial={graph_rate:.1%}{gold} | Oracle 행동={distribution}"
    )
    return {
        "direct_em": direct_em,
        "direct_f1": direct_f1,
        "oracle_em": oracle_em,
        "oracle_f1": oracle_f1,
    }


def _report_macro(rows: list[dict], datasets: Iterable[str]) -> None:
    groups = [[row for row in rows if row["dataset"] == dataset] for dataset in datasets]
    values = [_report_values(group) for group in groups if group]
    if values:
        _log(
            "Macro: "
            f"Direct EM/F1={np.mean([value['direct_em'] for value in values]):.3f}/"
            f"{np.mean([value['direct_f1'] for value in values]):.3f} | "
            f"Oracle EM/F1={np.mean([value['oracle_em'] for value in values]):.3f}/"
            f"{np.mean([value['oracle_f1'] for value in values]):.3f}"
        )


def _report_values(rows: list[dict]) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in ("direct_em", "direct_f1", "oracle_em", "oracle_f1")
    }


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
