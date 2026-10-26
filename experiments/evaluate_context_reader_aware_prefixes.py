"""Generate reader-aware search prefixes in each question's official context.

This is the benchmark-faithful counterpart of the earlier pooled-corpus pilot.
Questions are sampled only for ``train``; ``--max-queries-per-dataset 0``
streams the complete validation split.  Every output is resumable and gold
answers/supports are used only to create offline evaluation targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.evaluate_context_answer_oracle import _example_stream, _local_retriever
from experiments.train_adaptive_proof_completion_policy import ACTIONS, _support_metrics
from experiments.train_multihop_policy import (
    QUERY_SAMPLING_VERSION,
    SUPPORTED_DATASETS,
    MultiHopExample,
    _dataset_seed,
    _duration,
    _iter_jsonl,
    _normalise_processed_record,
    _reservoir_sample,
)
from src.planned_retrieval import run_planned_retrieval
from src.qa_evaluation import (
    EVIDENCE_CHAIN_PROMPT_VERSION,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    answer_score,
    evidence_context,
    ollama_model_identity,
)
from src.query_rewriter import (
    FrozenOllamaQueryRewriter,
    LEGACY_PROMPT_VERSION,
    QueryRewriteCache,
    QueryRewriterConfig,
)
from src.reader_sufficiency import (
    READER_SUFFICIENCY_FEATURE_NAMES,
    parse_evidence_chain,
    reader_sufficiency_features,
)
from src.settings import settings


FORMAT_VERSION = 1
CONTEXT_MODE = "official_per_question_distractor_reader_aware_v1"
CONTEXT_PREFIX_FEATURE_NAMES = (
    "search_step",
    "candidate_count",
    "retrieval_score_max",
    "retrieval_score_mean",
    "retrieval_score_min",
    "retrieval_score_gap_1_2",
    "new_candidate_fraction",
    "previous_candidate_survival_fraction",
    "planner_sufficient_known",
    "planner_reported_sufficient",
    "planner_missing_information_length",
    "planner_consecutive_no_new",
    *READER_SUFFICIENCY_FEATURE_NAMES,
)


def _sanitize_unicode(value):
    """Replace lone UTF-16 surrogates while preserving valid Unicode text."""
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, dict):
        return {
            _sanitize_unicode(key): _sanitize_unicode(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_unicode(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_unicode(item) for item in value)
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-context reader-aware prefix generation")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--max-queries-per-dataset", type=int, default=0)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument(
        "--exclude-ids-root",
        default="",
        help="train 표본과 겹치지 않도록 *_train.jsonl의 ID를 제외하고 새 표본을 추출합니다.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--answer-cache", default="data/benchmarks/multihop/context_reader_aware_answer_cache.sqlite3")
    parser.add_argument("--query-cache", default="data/benchmarks/multihop/context_reader_aware_query_cache.sqlite3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--initial-candidates", type=int, default=15)
    parser.add_argument("--max-search-steps", type=int, default=3)
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--planner-context-characters", type=int, default=4000)
    parser.add_argument("--anchor-slots", type=int, default=2)
    parser.add_argument("--reader-source-cap", type=int, default=8)
    parser.add_argument("--additions-per-step", type=int, default=1)
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--planner-max-tokens", type=int, default=80)
    parser.add_argument("--reader-max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_queries_per_dataset < 0:
        parser.error("max queries는 0 이상이어야 합니다.")
    if args.exclude_ids_root and (args.split != "train" or args.max_queries_per_dataset == 0):
        parser.error("exclude-ids-root는 유한 크기의 train 표본에서만 지원합니다.")
    if min(
        args.top_k,
        args.initial_candidates,
        args.max_search_steps,
        args.search_candidates,
        args.planner_context_characters,
        args.reader_source_cap,
        args.context_character_budget,
        args.planner_max_tokens,
        args.reader_max_tokens,
        args.timeout,
        args.log_every,
    ) < 1:
        parser.error("검색·문맥·token·log 설정은 1 이상이어야 합니다.")
    if args.initial_candidates < args.top_k or args.reader_source_cap < args.top_k:
        parser.error("initial candidates와 reader source cap은 top-k 이상이어야 합니다.")
    if args.max_search_steps != len(ACTIONS) - 1:
        parser.error(f"현재 정책 action schema는 추가 검색 {len(ACTIONS) - 1}회를 요구합니다.")
    return args


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_matches(path: Path, expected: Mapping) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        ignored = {"status", "completed_rows"}
        return all(saved.get(key) == value for key, value in expected.items() if key not in ignored)
    except (OSError, ValueError, TypeError):
        return False


def _write_manifest(path: Path, config: Mapping, status: str, completed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {**config, "status": status, "completed_rows": completed},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _holdout_stream(
    input_path: Path, dataset: str, maximum: int, seed: int, excluded: set[str]
) -> tuple[list[MultiHopExample], int]:
    records = (
        record
        for record in _iter_jsonl(input_path)
        if f"{dataset}:{record.get('id', '')}" not in excluded
    )
    sampled = _reservoir_sample(records, maximum, seed)
    examples = [_normalise_processed_record(record) for record in sampled]
    selected = sorted(
        (example for example in examples if example is not None),
        key=lambda example: example.policy_id,
    )
    if len(selected) != maximum:
        raise SystemExit(f"{dataset}: 제외 후 유효한 holdout 문항이 {maximum:,}개보다 적습니다.")
    if {example.policy_id for example in selected} & excluded:
        raise RuntimeError("holdout과 학습 표본의 ID가 겹칩니다.")
    return selected, len(selected)


def _candidate_rows(sources) -> list[dict]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "content_id": chunk.source,
            "title": chunk.title,
            "score": float(score),
        }
        for chunk, score in sources
    ]


def _retrieval_features(
    search_step: int,
    sources,
    previous_ids: set[str],
    planner_step,
) -> np.ndarray:
    scores = sorted((float(score) for _, score in sources), reverse=True)
    current_ids = {chunk.source for chunk, _ in sources}
    sufficient = planner_step.rewrite.sufficient if planner_step is not None else None
    missing = planner_step.rewrite.missing_information if planner_step is not None else ""
    consecutive_no_new = (
        planner_step.state.consecutive_no_new + int(not planner_step.added_chunks)
        if planner_step is not None
        else 0
    )
    return np.asarray(
        [
            float(search_step),
            float(len(sources)),
            max(scores, default=0.0),
            float(np.mean(scores)) if scores else 0.0,
            min(scores, default=0.0),
            scores[0] - scores[1] if len(scores) > 1 else max(scores, default=0.0),
            len(current_ids - previous_ids) / max(1, len(current_ids)),
            len(current_ids & previous_ids) / max(1, len(previous_ids)),
            float(sufficient is not None),
            float(sufficient is True),
            min(1.0, len(str(missing).split()) / 24.0),
            min(1.0, consecutive_no_new / 3.0),
        ],
        dtype="float64",
    )


def _evaluate_example(
    example: MultiHopExample,
    retriever,
    planner: FrozenOllamaQueryRewriter,
    reader: FrozenOllamaReader,
    args,
) -> dict:
    initial_started = perf_counter()
    initial = retriever.search(example.question, top_k=args.initial_candidates)
    initial_latency = perf_counter() - initial_started
    planned_started = perf_counter()
    planned = run_planned_retrieval(
        retriever,
        planner,
        example.question,
        initial,
        max_search_steps=args.max_search_steps,
        top_k=args.top_k,
        search_candidates=args.search_candidates,
        context_characters=args.planner_context_characters,
        anchor_slots=args.anchor_slots,
        reader_source_cap=args.reader_source_cap,
        additions_per_step=args.additions_per_step,
        replace_when_full=True,
    )
    planned_latency = perf_counter() - planned_started
    sources_by_step = {0: tuple(initial[: args.top_k]), **planned.sources_by_step}
    if any(step not in sources_by_step for step in range(args.max_search_steps + 1)):
        raise RuntimeError(f"검색 prefix가 완성되지 않았습니다: {example.policy_id}")

    documents = {document.content_id: document for document in example.documents}
    prior_predictions: list[str] = []
    prior_chains: list[tuple[str, ...]] = []
    previous_ids: set[str] = set()
    action_rows = {}
    cumulative_rewrite_latency = 0.0
    for search_step, action in enumerate(ACTIONS):
        sources = list(sources_by_step[search_step])
        planner_step = planned.steps[search_step - 1] if search_step else None
        if planner_step is not None:
            cumulative_rewrite_latency += float(planner_step.rewrite.latency_seconds)
        retrieval_features = _retrieval_features(
            search_step, sources, previous_ids, planner_step
        )
        context, provenance = evidence_context(sources, args.context_character_budget)
        result = reader.answer(example.question, context)
        chain = parse_evidence_chain(result.raw_response)
        reader_features = reader_sufficiency_features(
            example.question,
            result.prediction,
            chain,
            tuple({"title": chunk.title, "text": chunk.text} for chunk, _ in sources),
            prior_predictions,
            prior_chains,
        )
        policy_features = np.concatenate((retrieval_features, reader_features))
        if len(policy_features) != len(CONTEXT_PREFIX_FEATURE_NAMES):
            raise RuntimeError("official-context reader feature schema mismatch")
        score = answer_score(
            result.prediction,
            str(example.answer),
            example.answer_aliases,
            dataset=example.dataset,
        )
        visible_ids = {str(item["content_id"]) for item in provenance}
        recall, complete, precision = _support_metrics(visible_ids, example.supporting_groups)
        action_rows[action] = {
            "search_step": search_step,
            "prediction": result.prediction,
            "evidence_chain": list(chain),
            "em": score.em,
            "f1": score.f1,
            "source_recall": recall,
            "all_support_recall": complete,
            "support_precision": precision,
            "policy_features": policy_features.tolist(),
            "evidence": list(provenance),
            "candidate_evidence": _candidate_rows(sources),
            "context_characters": len(context),
            "reader_cache_hit": result.cache_hit,
            "reader_latency_seconds": result.latency_seconds,
            "reader_prompt_tokens": result.prompt_eval_count,
            "reader_generated_tokens": result.eval_count,
            "planner_rewrite_cache_hit": planner_step.rewrite.cache_hit if planner_step else None,
            "planner_rewrite_latency_cumulative_seconds": cumulative_rewrite_latency,
            "planner_reported_sufficient": planner_step.rewrite.sufficient if planner_step else None,
        }
        previous_ids = {chunk.source for chunk, _ in sources}
        prior_predictions.append(result.prediction)
        prior_chains.append(chain)

    return {
        "format_version": FORMAT_VERSION,
        "id": example.policy_id,
        "dataset": example.dataset,
        "question": example.question,
        "reasoning_type": example.reasoning_type,
        "hop_count": example.hop_count,
        "initial_retrieval_latency_seconds": initial_latency,
        "planner_total_latency_seconds": planned_latency,
        "planner_rewrite_calls": len(planned.steps),
        "actions": action_rows,
    }


def _run_dataset(
    args,
    dataset: str,
    examples: Iterable[MultiHopExample],
    total: int,
    planner,
    reader,
    planner_config,
    reader_config,
    embedding_model,
    reranker,
):
    input_path = Path(args.processed_root) / dataset / f"{args.split}.jsonl"
    output = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
    manifest = output.with_suffix(".manifest.json")
    config = {
        "format_version": FORMAT_VERSION,
        "experiment": "official_context_reader_aware_prefixes",
        "context_mode": CONTEXT_MODE,
        "dataset": dataset,
        "split": args.split,
        "processed_input": str(input_path),
        "processed_fingerprint": _file_fingerprint(input_path),
        "query_seed": args.query_seed,
        "query_sampling_version": QUERY_SAMPLING_VERSION,
        "max_queries_per_dataset": args.max_queries_per_dataset,
        "requested_rows": total,
        "actions": list(ACTIONS),
        "top_k": args.top_k,
        "initial_candidates": args.initial_candidates,
        "max_search_steps": args.max_search_steps,
        "search_candidates": args.search_candidates,
        "planner_context_characters": args.planner_context_characters,
        "anchor_slots": args.anchor_slots,
        "reader_source_cap": args.reader_source_cap,
        "additions_per_step": args.additions_per_step,
        "context_character_budget": args.context_character_budget,
        "feature_names": list(CONTEXT_PREFIX_FEATURE_NAMES),
        "planner": asdict(planner_config),
        "reader": asdict(reader_config),
    }
    if args.exclude_ids_root:
        excluded = args.excluded_ids_by_dataset[dataset]
        config.update(
            {
                "exclude_ids_root": args.exclude_ids_root,
                "exclude_count": len(excluded),
                "exclude_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(excluded)).encode("utf-8")
                ).hexdigest(),
                "selection_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(example.policy_id for example in examples)).encode("utf-8")
                ).hexdigest(),
            }
        )
    cached = {}
    if output.exists() or manifest.exists():
        if args.overwrite:
            cached = {}
        elif _manifest_matches(manifest, config):
            cached = {str(row["id"]): row for row in _iter_jsonl(output)} if output.exists() else {}
        else:
            raise SystemExit(f"기존 official-context prefix 설정과 다릅니다: {output}")
    if len(cached) > total:
        raise SystemExit(f"캐시 행이 요청 문항보다 많습니다: {output}")
    if len(cached) == total:
        print(f"{dataset}: reader-aware prefix {total:,}개 전체 재사용", flush=True)
        return embedding_model, reranker

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output.exists() else "a"
    if mode == "w":
        cached = {}
    _write_manifest(manifest, config, "in_progress", len(cached))
    started = perf_counter()
    new = 0
    terminal_em = sum(float(row["actions"][ACTIONS[-1]]["em"]) for row in cached.values())
    terminal_f1 = sum(float(row["actions"][ACTIONS[-1]]["f1"]) for row in cached.values())
    last_retriever = None
    with output.open(mode, encoding="utf-8") as destination:
        for example in examples:
            if example.policy_id in cached:
                continue
            retriever = _local_retriever(
                example,
                embedding_model=embedding_model,
                reranker=reranker,
                semantic_neighbors=3,
            )
            row = _evaluate_example(example, retriever, planner, reader, args)
            row = _sanitize_unicode(row)
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            cached[example.policy_id] = row
            terminal_em += float(row["actions"][ACTIONS[-1]]["em"])
            terminal_f1 += float(row["actions"][ACTIONS[-1]]["f1"])
            new += 1
            embedding_model = retriever.model or embedding_model
            reranker = retriever.reranker or reranker
            last_retriever = retriever
            completed = len(cached)
            if completed % args.log_every == 0 or completed == total:
                elapsed = perf_counter() - started
                eta = elapsed / max(1, new) * max(0, total - completed)
                print(
                    f"{dataset} official reader-aware {completed:,}/{total:,} "
                    f"({completed / total:.1%}) | ETA {_duration(eta)} | "
                    f"fixed3 EM/F1={terminal_em/completed*100:.2f}/{terminal_f1/completed*100:.2f}",
                    flush=True,
                )
                _write_manifest(manifest, config, "in_progress", completed)
    if len(cached) != total:
        raise SystemExit(f"{dataset}: 완료 행 {len(cached):,}개가 요청 {total:,}개와 다릅니다.")
    _write_manifest(manifest, config, "ready", len(cached))
    if last_retriever is not None:
        embedding_model = last_retriever.model or embedding_model
        reranker = last_retriever.reranker or reranker
    return embedding_model, reranker


def main() -> None:
    args = _arguments()
    args.excluded_ids_by_dataset = {}
    if args.exclude_ids_root:
        for dataset in args.datasets:
            exclude_path = Path(args.exclude_ids_root) / f"{dataset}_train.jsonl"
            if not exclude_path.exists():
                raise SystemExit(f"제외할 학습 표본이 없습니다: {exclude_path}")
            args.excluded_ids_by_dataset[dataset] = {
                str(row["id"]) for row in _iter_jsonl(exclude_path)
            }
    session = requests.Session()
    model, digest = ollama_model_identity(args.ollama_host, args.model, session=session)
    planner_config = QueryRewriterConfig(
        host=args.ollama_host,
        model=model,
        model_digest=digest,
        max_tokens=args.planner_max_tokens,
        timeout_seconds=args.timeout,
        seed=args.model_seed,
        temperature=0.0,
        keep_alive=args.ollama_keep_alive,
        prompt_version=LEGACY_PROMPT_VERSION,
    )
    reader_config = ReaderConfig(
        host=args.ollama_host,
        model=model,
        model_digest=digest,
        max_tokens=args.reader_max_tokens,
        timeout_seconds=args.timeout,
        seed=args.model_seed,
        temperature=0.0,
        keep_alive=args.ollama_keep_alive,
        prompt_version=EVIDENCE_CHAIN_PROMPT_VERSION,
        thinking=False,
    )
    print(
        f"공식 context reader-aware prefix 시작: split={args.split}, "
        f"sample={args.max_queries_per_dataset or '전체'}, query_seed={args.query_seed}, "
        f"model={model}",
        flush=True,
    )
    embedding_model = reranker = None
    with ExitStack() as stack:
        answer_cache = stack.enter_context(AnswerCache(Path(args.answer_cache)))
        query_cache = stack.enter_context(QueryRewriteCache(Path(args.query_cache)))
        reader = FrozenOllamaReader(reader_config, answer_cache, session=session)
        planner = FrozenOllamaQueryRewriter(planner_config, query_cache, session=session)
        for dataset in args.datasets:
            input_path = Path(args.processed_root) / dataset / f"{args.split}.jsonl"
            if args.exclude_ids_root:
                examples, total = _holdout_stream(
                    input_path,
                    dataset,
                    args.max_queries_per_dataset,
                    _dataset_seed(args.query_seed, dataset),
                    args.excluded_ids_by_dataset[dataset],
                )
            else:
                examples, total = _example_stream(
                    input_path,
                    maximum=args.max_queries_per_dataset,
                    seed=_dataset_seed(args.query_seed, dataset),
                )
            embedding_model, reranker = _run_dataset(
                args,
                dataset,
                examples,
                total,
                planner,
                reader,
                planner_config,
                reader_config,
                embedding_model,
                reranker,
            )
    session.close()


if __name__ == "__main__":
    main()
