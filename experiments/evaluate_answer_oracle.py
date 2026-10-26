"""Measure answer-level retrieval-action potential with one frozen QA reader.

The command enumerates the current bounded Direct/Graph actions, gives every
terminal evidence set to the exact same deterministic reader, and reports the
Answer Oracle gap over Direct Hybrid Retrieval.  It is the go/no-go audit that
must precede a larger sequential-policy training run.

Example pilot:
    python -m experiments.evaluate_answer_oracle \
      --datasets hotpotqa musique 2wiki \
      --split train \
      --corpus-examples-per-dataset 2000 \
      --max-queries-per-dataset 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.train_multihop_policy import (
    QUERY_SAMPLING_VERSION,
    SUPPORTED_DATASETS,
    MultiHopExample,
    _corpus_fingerprint,
    _dataset_seed,
    _duration,
    _example_fingerprint,
    _iter_jsonl,
    _matches_index_manifest,
    load_processed_examples,
    retrieve_for_action,
    select_query_examples,
)
from src.policy import POLICY_FEATURE_NAMES, profile_features
from src.document_loader import Chunk
from src.qa_evaluation import (
    PROMPT_VERSION,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    answer_score,
    evidence_context,
    ollama_model_identity,
)
from src.retriever import SemanticRetriever
from src.routing import POLICY_ACTION_IDS, EvidenceProfile
from src.settings import settings


OUTPUT_FORMAT_VERSION = 2
SEQUENTIAL_OUTPUT_FORMAT_VERSION = 6
DIRECT_ACTION = "vector"
ACTION_LOG_NAMES = {
    "graph_1hop_small": "1-hop small",
    "graph_1hop_large": "1-hop large",
    "graph_2hop_small": "2-hop small",
    "graph_2hop_large": "2-hop large",
}
FIRST_STEP_ACTIONS = ("graph_1hop_small", "graph_1hop_large")
SECOND_STEP_ACTIONS = ("expand_2hop_small", "expand_2hop_large")
RETURN_DIRECT_ACTION = "return_direct"
POST_FIRST_STEP_ACTIONS = (RETURN_DIRECT_ACTION, *SECOND_STEP_ACTIONS)
SECOND_STEP_TERMINAL_ACTION = {
    "expand_2hop_small": "graph_2hop_small",
    "expand_2hop_large": "graph_2hop_large",
}
SEQUENTIAL_STATE_FEATURE_NAMES = (
    *(f"current_{name}" for name in POLICY_FEATURE_NAMES),
    *(f"delta_{name}" for name in POLICY_FEATURE_NAMES),
    "new_chunk_ratio",
    "new_source_ratio",
    "first_step_small",
    "first_step_large",
    "first_step_extra_candidates",
)


def main() -> None:
    args = _arguments()
    started = perf_counter()
    mode = "Sequential Answer Oracle" if args.sequential_trajectories else "Answer Oracle"
    _log(f"멀티홉 QA {mode} pilot을 시작합니다. QA reader는 동결되며 검색 행동만 달라집니다.")

    session = requests.Session()
    try:
        resolved_model, model_digest = ollama_model_identity(
            args.ollama_host,
            args.reader_model,
            session=session,
        )
    except RuntimeError as error:
        raise SystemExit(
            f"{error}\n먼저 'ollama serve'를 실행하고 'ollama list'로 reader 모델을 확인하세요."
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

    all_rows: list[dict] = []
    shared_embedding_model = None
    shared_reranker = None
    with AnswerCache(Path(args.cache_path)) as cache:
        reader = FrozenOllamaReader(reader_config, cache, session=session)
        for dataset in args.datasets:
            path = Path(args.processed_root) / dataset / f"{args.split}.jsonl"
            corpus_examples = load_processed_examples(
                path,
                args.corpus_examples_per_dataset,
                _dataset_seed(args.corpus_seed, dataset),
            )
            if not corpus_examples:
                raise SystemExit(f"유효한 전처리 문항이 없습니다: {path}")
            if args.max_queries_per_dataset and args.max_queries_per_dataset > len(corpus_examples):
                raise SystemExit(
                    f"{dataset}: 요청 query {args.max_queries_per_dataset:,}개가 corpus cohort "
                    f"{len(corpus_examples):,}개보다 많습니다."
                )
            examples = select_query_examples(
                corpus_examples,
                args.max_queries_per_dataset,
                _dataset_seed(args.query_seed, dataset),
            )
            missing_answers = [example.policy_id for example in examples if not example.answer]
            if missing_answers:
                raise SystemExit(f"{dataset}: 정답이 없는 query가 있습니다: {missing_answers[0]}")

            corpus_fingerprint = _corpus_fingerprint(corpus_examples)
            index_dir = Path(args.index_root) / args.split / dataset
            retriever = _load_fixed_retriever(
                index_dir,
                corpus_fingerprint,
                embedding_model=shared_embedding_model,
                reranker=shared_reranker,
            )
            shared_embedding_model = retriever.model or shared_embedding_model
            shared_reranker = retriever.reranker or shared_reranker

            output_suffix = "_sequential_v6" if args.sequential_trajectories else ""
            output_path = Path(args.output_root) / f"{dataset}_{args.split}{output_suffix}.jsonl"
            config = _output_config(
                args,
                dataset=dataset,
                corpus_fingerprint=corpus_fingerprint,
                reader_config=reader_config,
            )
            rows = _evaluate_dataset(
                retriever,
                reader,
                examples,
                output_path,
                config,
                actions=args.actions,
                top_k=args.top_k,
                context_character_budget=args.context_character_budget,
                log_every=args.log_every,
                overwrite=args.overwrite,
                sequential_trajectories=args.sequential_trajectories,
            )
            all_rows.extend(rows)
            _report_group(rows, dataset)
            shared_embedding_model = retriever.model or shared_embedding_model
            shared_reranker = retriever.reranker or shared_reranker

    _report_group(all_rows, "전체 Micro")
    _report_macro(all_rows, args.datasets)
    _log(f"{mode} pilot 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="고정 reader 기반 멀티홉 QA Answer Oracle 평가")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--output-root", default="data/benchmarks/multihop/answer_oracle")
    parser.add_argument("--cache-path", default="data/benchmarks/multihop/answer_cache.sqlite3")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument(
        "--max-queries-per-dataset",
        "--max-examples-per-dataset",
        dest="max_queries_per_dataset",
        type=int,
        default=10,
    )
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--actions", nargs="+", choices=POLICY_ACTION_IDS, default=list(POLICY_ACTION_IDS))
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--reader-model", default=settings.ollama_model)
    parser.add_argument("--reader-max-tokens", type=int, default=64)
    parser.add_argument("--reader-timeout", type=int, default=300)
    parser.add_argument("--reader-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--sequential-trajectories",
        action="store_true",
        help="1-hop 결과를 재관찰한 뒤 STOP/2-hop을 비교하는 순차 trajectory audit",
    )
    parser.add_argument("--overwrite", action="store_true", help="기존 결과 JSONL을 새로 생성")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    args.actions = tuple(dict.fromkeys(args.actions))
    if DIRECT_ACTION not in args.actions:
        parser.error("Answer Oracle gap 계산을 위해 vector(Direct Hybrid) 행동이 필요합니다.")
    if args.sequential_trajectories and not set(POLICY_ACTION_IDS).issubset(args.actions):
        parser.error(
            "순차 trajectory의 공정한 terminal-action 비교에는 vector와 네 Graph 행동이 모두 필요합니다."
        )
    if args.corpus_examples_per_dataset < 0 or args.max_queries_per_dataset < 0:
        parser.error("corpus/query 수는 0 이상이어야 합니다.")
    if args.top_k < 1 or args.context_character_budget < 1 or args.log_every < 1:
        parser.error("top-k, context budget, log 간격은 1 이상이어야 합니다.")
    if args.reader_max_tokens < 1 or args.reader_timeout < 1:
        parser.error("reader token/timeout은 1 이상이어야 합니다.")
    return args


def _load_fixed_retriever(
    index_dir: Path,
    corpus_fingerprint: str,
    *,
    embedding_model=None,
    reranker=None,
) -> SemanticRetriever:
    manifest_path = index_dir / "multihop_policy_manifest.json"
    if not _matches_index_manifest(manifest_path, corpus_fingerprint):
        raise SystemExit(
            f"고정 corpus 인덱스가 없거나 설정이 다릅니다: {index_dir}\n"
            "먼저 train_multihop_policy --prepare-index-only를 실행하세요."
        )
    retriever = SemanticRetriever(index_dir)
    retriever.model = embedding_model
    retriever.reranker = reranker
    if not retriever.load():
        raise SystemExit(f"고정 corpus 인덱스를 읽지 못했습니다: {index_dir}")
    _log(f"고정 corpus 인덱스 로드: {index_dir} · 문서 청크 {len(retriever.chunks):,}개")
    return retriever


def _evaluate_dataset(
    retriever: SemanticRetriever,
    reader: FrozenOllamaReader,
    examples: list[MultiHopExample],
    output_path: Path,
    config: Mapping,
    *,
    actions: tuple[str, ...],
    top_k: int,
    context_character_budget: int,
    log_every: int,
    overwrite: bool,
    sequential_trajectories: bool = False,
) -> list[dict]:
    manifest_path = output_path.with_suffix(".manifest.json")
    cached_rows: dict[str, dict] = {}
    if output_path.exists() or manifest_path.exists():
        if overwrite:
            cached_rows = {}
        elif _matching_manifest(manifest_path, config):
            cached_rows = {row["id"]: row for row in _iter_jsonl(output_path)} if output_path.exists() else {}
        else:
            raise SystemExit(
                f"기존 Answer Oracle 결과 설정이 현재 실행과 다릅니다: {output_path}\n"
                "다른 --output-root를 사용하거나 --overwrite를 지정하세요."
            )

    expected_ids = {example.policy_id for example in examples}
    selected_rows = {row_id: row for row_id, row in cached_rows.items() if row_id in expected_ids}
    if len(selected_rows) == len(examples):
        _log(
            f"{examples[0].dataset}: Answer 결과 {len(selected_rows):,}개 재사용 "
            f"(전체 결과 cache {len(cached_rows):,}개)"
        )
        return [selected_rows[example.policy_id] for example in examples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not cached_rows:
        cached_rows = {}
        selected_rows = {}
        mode = "w"
    else:
        mode = "a"
    _write_manifest(
        manifest_path,
        config,
        status="in_progress",
        cached_rows=len(cached_rows),
        requested_rows=len(examples),
        requested_completed=len(selected_rows),
    )

    started = perf_counter()
    newly_completed = 0
    with output_path.open(mode, encoding="utf-8") as destination:
        for example in examples:
            if example.policy_id in selected_rows:
                continue
            row = _evaluate_example(
                retriever,
                reader,
                example,
                actions=actions,
                top_k=top_k,
                context_character_budget=context_character_budget,
                sequential_trajectories=sequential_trajectories,
            )
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            cached_rows[example.policy_id] = row
            selected_rows[example.policy_id] = row
            newly_completed += 1
            completed = len(selected_rows)
            if completed % log_every == 0 or completed == len(examples):
                elapsed = perf_counter() - started
                eta = elapsed / max(1, newly_completed) * (len(examples) - completed)
                current_rows = list(selected_rows.values())
                cache_calls = sum(
                    int(action["reader_cache_hit"])
                    for saved in current_rows
                    for action in saved["actions"].values()
                ) + sum(int(saved["gold_support"]["reader_cache_hit"]) for saved in current_rows)
                total_calls = sum(len(saved["actions"]) + 1 for saved in current_rows)
                oracle_key = "sequential_oracle_f1" if sequential_trajectories else "oracle_f1"
                _log(
                    f"{example.dataset} Answer {completed:,}/{len(examples):,} ({completed / len(examples):.1%}) "
                    f"| ETA {_duration(eta)} | Direct F1={_average(current_rows, DIRECT_ACTION, 'f1'):.3f} "
                    f"| {'Sequential ' if sequential_trajectories else ''}Oracle F1="
                    f"{float(np.mean([saved[oracle_key] for saved in current_rows])):.3f} "
                    f"| reader cache hit={cache_calls / max(1, total_calls):.0%}"
                )
                _write_manifest(
                    manifest_path,
                    config,
                    status="in_progress",
                    cached_rows=len(cached_rows),
                    requested_rows=len(examples),
                    requested_completed=completed,
                )

    _write_manifest(
        manifest_path,
        config,
        status="ready",
        cached_rows=len(cached_rows),
        requested_rows=len(examples),
        requested_completed=len(selected_rows),
    )
    return [selected_rows[example.policy_id] for example in examples]


def _evaluate_example(
    retriever: SemanticRetriever,
    reader: FrozenOllamaReader,
    example: MultiHopExample,
    *,
    actions: tuple[str, ...],
    top_k: int,
    context_character_budget: int,
    sequential_trajectories: bool = False,
    initial_candidates: int | None = None,
    include_gold_support: bool = True,
) -> dict:
    if example.answer is None:
        raise ValueError(f"정답이 없는 example입니다: {example.policy_id}")
    retrieval_started = perf_counter()
    initial_count = initial_candidates if initial_candidates is not None else max(top_k * 3, 12)
    if initial_count < top_k:
        raise ValueError("initial candidates는 top-k 이상이어야 합니다.")
    initial = retriever.search(example.question, top_k=initial_count)
    initial_latency = perf_counter() - retrieval_started
    profile = retriever._evidence_profile(initial)
    action_results: dict[str, dict] = {}
    sources_by_action: dict[str, list[tuple[Chunk, float]]] = {}

    for action_id in actions:
        action_started = perf_counter()
        sources, extra_candidates = retrieve_for_action(
            retriever,
            example.question,
            initial,
            profile,
            action_id,
            top_k,
        )
        action_latency = perf_counter() - action_started
        sources_by_action[action_id] = sources
        action_results[action_id] = _score_sources(
            reader,
            example,
            sources,
            extra_candidates=extra_candidates,
            initial_retrieval_latency=initial_latency,
            action_retrieval_latency=action_latency,
            context_character_budget=context_character_budget,
        )

    oracle_action = _best_answer_action(action_results)
    direct = action_results[DIRECT_ACTION]
    oracle = action_results[oracle_action]
    row = {
        "format_version": (
            SEQUENTIAL_OUTPUT_FORMAT_VERSION if sequential_trajectories else OUTPUT_FORMAT_VERSION
        ),
        "id": example.policy_id,
        "example_id": example.example_id,
        "dataset": example.dataset,
        "question": example.question,
        "answer": example.answer,
        "answer_aliases": list(example.answer_aliases),
        "hop_count": example.hop_count,
        "reasoning_type": example.reasoning_type,
        "features": dict(zip(POLICY_FEATURE_NAMES, profile_features(profile).tolist(), strict=True)),
        "actions": action_results,
        "oracle_action": oracle_action,
        "direct_em": direct["em"],
        "direct_f1": direct["f1"],
        "oracle_em": oracle["em"],
        "oracle_f1": oracle["f1"],
        "oracle_em_gain": oracle["em"] - direct["em"],
        "oracle_f1_gain": oracle["f1"] - direct["f1"],
    }
    if include_gold_support:
        gold_sources = _gold_support_sources(
            retriever,
            example.question,
            example.supporting_groups,
            top_k,
        )
        gold_context, gold_provenance = evidence_context(gold_sources, context_character_budget)
        gold_reader_result = reader.answer(example.question, gold_context)
        gold_score = answer_score(
            gold_reader_result.prediction,
            example.answer,
            example.answer_aliases,
            dataset=example.dataset,
        )
        gold_visible_sources = {str(item["content_id"]) for item in gold_provenance}
        gold_recovered = sum(bool(gold_visible_sources & group) for group in example.supporting_groups)
        gold_support_result = {
            "prediction": gold_reader_result.prediction,
            "raw_response": gold_reader_result.raw_response,
            "matched_gold": gold_score.matched_gold,
            "em": gold_score.em,
            "f1": gold_score.f1,
            "source_recall": gold_recovered / len(example.supporting_groups),
            "all_support_recall": float(gold_recovered == len(example.supporting_groups)),
            "reader_latency_seconds": gold_reader_result.latency_seconds,
            "reader_cache_hit": gold_reader_result.cache_hit,
            "prompt_eval_count": gold_reader_result.prompt_eval_count,
            "eval_count": gold_reader_result.eval_count,
            "context_hash": hashlib.sha256(gold_context.encode("utf-8")).hexdigest(),
            "context_characters": len(gold_context),
            "evidence": list(gold_provenance),
        }
        row["gold_support"] = gold_support_result
        row["gold_support_em"] = gold_support_result["em"]
        row["gold_support_f1"] = gold_support_result["f1"]
    if sequential_trajectories:
        missing = [action for action in POLICY_ACTION_IDS if action not in sources_by_action]
        if missing:
            raise ValueError(f"순차 trajectory에 필요한 행동이 없습니다: {missing}")
        sequential = _evaluate_sequential_trajectories(
            retriever,
            initial_profile=profile,
            initial_sources=sources_by_action[DIRECT_ACTION],
            first_sources={action: sources_by_action[action] for action in FIRST_STEP_ACTIONS},
            action_results=action_results,
        )
        row["sequential"] = sequential
        row["sequential_oracle_trajectory"] = sequential["oracle_trajectory"]
        row["sequential_oracle_em"] = sequential["oracle_em"]
        row["sequential_oracle_f1"] = sequential["oracle_f1"]
        row["sequential_oracle_em_gain"] = sequential["oracle_em"] - direct["em"]
        row["sequential_oracle_f1_gain"] = sequential["oracle_f1"] - direct["f1"]
    return row


def _score_sources(
    reader: FrozenOllamaReader,
    example: MultiHopExample,
    sources: list[tuple[Chunk, float]],
    *,
    extra_candidates: int,
    initial_retrieval_latency: float,
    action_retrieval_latency: float,
    context_character_budget: int,
    context_allocation: str = "sequential",
    context_compression: str = "none",
    focus_queries: tuple[str, ...] = (),
    focused_sentences_per_source: int = 2,
) -> dict:
    if example.answer is None:
        raise ValueError(f"정답이 없는 example입니다: {example.policy_id}")
    context, provenance = evidence_context(
        sources,
        context_character_budget,
        allocation=context_allocation,
        compression=context_compression,
        focus_queries=(example.question, *focus_queries),
        focused_sentences_per_source=focused_sentences_per_source,
    )
    result = reader.answer(example.question, context)
    score = answer_score(
        result.prediction,
        example.answer,
        example.answer_aliases,
        dataset=example.dataset,
    )
    retrieved_sources = {str(item["content_id"]) for item in provenance}
    recovered = sum(bool(retrieved_sources & group) for group in example.supporting_groups)
    source_recall = recovered / len(example.supporting_groups) if example.supporting_groups else 0.0
    all_support = float(recovered == len(example.supporting_groups)) if example.supporting_groups else 0.0
    return {
        "prediction": result.prediction,
        "raw_response": result.raw_response,
        "matched_gold": score.matched_gold,
        "em": score.em,
        "f1": score.f1,
        "source_recall": source_recall,
        "all_support_recall": all_support,
        "extra_candidates": extra_candidates,
        "initial_retrieval_latency_seconds": initial_retrieval_latency,
        "action_retrieval_latency_seconds": action_retrieval_latency,
        "reader_latency_seconds": result.latency_seconds,
        "reader_cache_hit": result.cache_hit,
        "prompt_eval_count": result.prompt_eval_count,
        "eval_count": result.eval_count,
        "context_hash": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "context_characters": len(context),
        "evidence": list(provenance),
    }


def _evaluate_sequential_trajectories(
    retriever: SemanticRetriever,
    *,
    initial_profile: EvidenceProfile,
    initial_sources: list[tuple[Chunk, float]],
    first_sources: Mapping[str, list[tuple[Chunk, float]]],
    action_results: Mapping[str, dict],
) -> dict:
    """Build sequential decisions over the same terminal evidence as one-shot.

    Sequential routing is allowed to observe a real 1-hop state before deciding
    whether to stop or continue.  It is *not* allowed a different terminal
    retrieval space: a 2-hop continuation reuses the corresponding one-shot
    graph_2hop evidence/result.  Consequently both policies have the same
    answer Oracle; their learned-policy regret is the meaningful comparison.
    """
    direct_result = action_results[DIRECT_ACTION]
    terminal_results: dict[str, dict] = {"stop": direct_result}
    second_step_results: dict[str, dict] = {}
    states: dict[str, dict] = {}

    for first_action in FIRST_STEP_ACTIONS:
        current_sources = first_sources[first_action]
        current_result = action_results[first_action]
        current_profile = retriever._evidence_profile(current_sources)
        state_values = _sequential_state_features(
            initial_profile,
            current_profile,
            initial_sources,
            current_sources,
            first_action=first_action,
            first_extra_candidates=int(current_result["extra_candidates"]),
        )
        first_trajectory = f"{first_action}_stop"
        terminal_results[first_trajectory] = current_result
        state_action_trajectories = {"stop": first_trajectory}

        return_trajectory = f"{first_action}_{RETURN_DIRECT_ACTION}"
        return_result = dict(direct_result)
        return_result["terminal_action"] = DIRECT_ACTION
        return_result["extra_candidates"] = int(current_result["extra_candidates"])
        return_result["action_retrieval_latency_seconds"] = float(
            current_result["action_retrieval_latency_seconds"]
        )
        return_result["returned_after_first_action"] = first_action
        terminal_results[return_trajectory] = return_result
        state_action_trajectories[RETURN_DIRECT_ACTION] = return_trajectory

        for second_action in SECOND_STEP_ACTIONS:
            terminal_action = SECOND_STEP_TERMINAL_ACTION[second_action]
            terminal_action_result = action_results[terminal_action]
            second_extra = int(terminal_action_result["extra_candidates"])
            cumulative_extra = int(current_result["extra_candidates"]) + second_extra
            trajectory = f"{first_action}_{second_action}"
            result = dict(terminal_action_result)
            result["terminal_action"] = terminal_action
            result["extra_candidates"] = cumulative_extra
            result["action_retrieval_latency_seconds"] = (
                float(current_result["action_retrieval_latency_seconds"])
                + float(terminal_action_result["action_retrieval_latency_seconds"])
            )
            result["second_step_extra_candidates"] = second_extra
            terminal_results[trajectory] = result
            second_step_results[trajectory] = result
            state_action_trajectories[second_action] = trajectory

        state_oracle_action = _best_state_action(state_action_trajectories, terminal_results)
        states[first_action] = {
            "feature_names": list(SEQUENTIAL_STATE_FEATURE_NAMES),
            "features": dict(zip(SEQUENTIAL_STATE_FEATURE_NAMES, state_values.tolist(), strict=True)),
            "action_trajectories": state_action_trajectories,
            "action_values": {
                action: {
                    "f1": terminal_results[trajectory]["f1"],
                    "em": terminal_results[trajectory]["em"],
                    "extra_candidates": terminal_results[trajectory]["extra_candidates"],
                }
                for action, trajectory in state_action_trajectories.items()
            },
            "oracle_action": state_oracle_action,
            "oracle_trajectory": state_action_trajectories[state_oracle_action],
        }

    s0_action_trajectories = {"stop": "stop"}
    for first_action in FIRST_STEP_ACTIONS:
        s0_action_trajectories[first_action] = states[first_action]["oracle_trajectory"]
    s0_oracle_action = _best_state_action(s0_action_trajectories, terminal_results)
    oracle_trajectory = _best_trajectory(terminal_results)
    oracle = terminal_results[oracle_trajectory]
    one_shot_oracle = action_results[_best_answer_action(action_results)]
    if not (
        np.isclose(float(oracle["f1"]), float(one_shot_oracle["f1"]))
        and np.isclose(float(oracle["em"]), float(one_shot_oracle["em"]))
    ):
        raise AssertionError("Sequential과 one-shot의 공유 terminal Oracle이 일치하지 않습니다.")
    return {
        "state_feature_names": list(SEQUENTIAL_STATE_FEATURE_NAMES),
        "states_after_first_step": states,
        "second_step_results": second_step_results,
        "terminal_results": terminal_results,
        "s0_action_trajectories": s0_action_trajectories,
        "s0_action_values": {
            action: {
                "f1": terminal_results[trajectory]["f1"],
                "em": terminal_results[trajectory]["em"],
                "extra_candidates": terminal_results[trajectory]["extra_candidates"],
            }
            for action, trajectory in s0_action_trajectories.items()
        },
        "s0_oracle_action": s0_oracle_action,
        "oracle_trajectory": oracle_trajectory,
        "oracle_em": oracle["em"],
        "oracle_f1": oracle["f1"],
        "shared_terminal_oracle": True,
    }


def _sequential_state_features(
    initial_profile: EvidenceProfile,
    current_profile: EvidenceProfile,
    initial_sources: list[tuple[Chunk, float]],
    current_sources: list[tuple[Chunk, float]],
    *,
    first_action: str,
    first_extra_candidates: int,
) -> np.ndarray:
    initial_values = profile_features(initial_profile)
    current_values = profile_features(current_profile)
    initial_chunk_ids = {chunk.chunk_id for chunk, _ in initial_sources}
    current_chunk_ids = {chunk.chunk_id for chunk, _ in current_sources}
    initial_source_ids = {chunk.source for chunk, _ in initial_sources}
    current_source_ids = {chunk.source for chunk, _ in current_sources}
    return np.concatenate(
        (
            current_values,
            current_values - initial_values,
            np.asarray(
                (
                    len(current_chunk_ids - initial_chunk_ids) / max(1, len(current_chunk_ids)),
                    len(current_source_ids - initial_source_ids) / max(1, len(current_source_ids)),
                    float(first_action == "graph_1hop_small"),
                    float(first_action == "graph_1hop_large"),
                    min(1.0, first_extra_candidates / max(1, settings.graph_large_budget)),
                ),
                dtype="float64",
            ),
        )
    )


def _best_state_action(
    action_trajectories: Mapping[str, str],
    terminal_results: Mapping[str, Mapping],
) -> str:
    actions = tuple(action_trajectories)
    return max(
        actions,
        key=lambda action: (
            float(terminal_results[action_trajectories[action]]["f1"]),
            float(terminal_results[action_trajectories[action]]["em"]),
            -int(terminal_results[action_trajectories[action]]["extra_candidates"]),
            -actions.index(action),
        ),
    )


def _best_trajectory(terminal_results: Mapping[str, Mapping]) -> str:
    trajectories = tuple(terminal_results)
    return max(
        trajectories,
        key=lambda trajectory: (
            float(terminal_results[trajectory]["f1"]),
            float(terminal_results[trajectory]["em"]),
            -int(terminal_results[trajectory]["extra_candidates"]),
            -trajectories.index(trajectory),
        ),
    )


def _gold_support_sources(
    retriever: SemanticRetriever,
    question: str,
    supporting_groups: tuple[frozenset[str], ...],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    """Build a same-budget reader ceiling from gold support documents.

    One highest-scoring chunk is reserved for every gold support group before
    remaining gold chunks fill the shared Top-K budget. Duplicate-title groups
    therefore contribute one document rather than being counted repeatedly.
    """
    if len(supporting_groups) > top_k:
        raise ValueError(
            f"gold support 그룹 {len(supporting_groups)}개가 reader Top-K {top_k}를 초과합니다."
        )
    allowed_sources = set().union(*supporting_groups) if supporting_groups else set()
    candidates = [(chunk, 1.0) for chunk in retriever.chunks if chunk.source in allowed_sources]
    ranked = retriever._rerank(question, candidates)
    selected: list[tuple[Chunk, float]] = []
    selected_chunk_ids: set[str] = set()
    for group in supporting_groups:
        best = next((item for item in ranked if item[0].source in group), None)
        if best is not None and best[0].chunk_id not in selected_chunk_ids:
            selected.append(best)
            selected_chunk_ids.add(best[0].chunk_id)
    for item in ranked:
        if len(selected) >= top_k:
            break
        if item[0].chunk_id not in selected_chunk_ids:
            selected.append(item)
            selected_chunk_ids.add(item[0].chunk_id)
    return sorted(selected, key=lambda item: item[1], reverse=True)


def _best_answer_action(actions: Mapping[str, Mapping]) -> str:
    available = tuple(action for action in POLICY_ACTION_IDS if action in actions)
    return max(
        available,
        key=lambda action: (
            float(actions[action]["f1"]),
            float(actions[action]["em"]),
            -int(actions[action]["extra_candidates"]),
            -POLICY_ACTION_IDS.index(action),
        ),
    )


def _output_config(
    args: argparse.Namespace,
    *,
    dataset: str,
    corpus_fingerprint: str,
    reader_config: ReaderConfig,
) -> dict:
    payload = {
        "format_version": (
            SEQUENTIAL_OUTPUT_FORMAT_VERSION
            if args.sequential_trajectories
            else OUTPUT_FORMAT_VERSION
        ),
        "dataset": dataset,
        "split": args.split,
        "corpus_fingerprint": corpus_fingerprint,
        "corpus_seed": args.corpus_seed,
        "query_seed": args.query_seed,
        "query_sampling_version": QUERY_SAMPLING_VERSION,
        "top_k": args.top_k,
        "context_character_budget": args.context_character_budget,
        "actions": list(args.actions),
        "reader": asdict(reader_config),
    }
    if args.sequential_trajectories:
        payload.update(
            {
                "sequential_trajectories": True,
                "sequential_state_feature_names": list(SEQUENTIAL_STATE_FEATURE_NAMES),
            }
        )
    return payload


def _matching_manifest(path: Path, expected: Mapping) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return all(saved.get(key) == value for key, value in expected.items())
    except (OSError, ValueError, TypeError):
        return False


def _write_manifest(
    path: Path,
    config: Mapping,
    *,
    status: str,
    cached_rows: int,
    requested_rows: int,
    requested_completed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(config)
    payload.update(
        {
            "status": status,
            "cached_rows": cached_rows,
            "last_requested_rows": requested_rows,
            "last_requested_completed": requested_completed,
        }
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _average(rows: list[dict], action: str, metric: str) -> float:
    return float(np.mean([float(row["actions"][action][metric]) for row in rows])) if rows else 0.0


def _report_group(rows: list[dict], name: str) -> dict[str, float]:
    if not rows:
        return {"direct_em": 0.0, "direct_f1": 0.0, "oracle_em": 0.0, "oracle_f1": 0.0}
    direct_em = _average(rows, DIRECT_ACTION, "em")
    direct_f1 = _average(rows, DIRECT_ACTION, "f1")
    oracle_em = float(np.mean([row["oracle_em"] for row in rows]))
    oracle_f1 = float(np.mean([row["oracle_f1"] for row in rows]))
    graph_beneficial = float(np.mean([row["oracle_f1_gain"] > 1e-9 for row in rows]))
    distribution = dict(sorted(Counter(row["oracle_action"] for row in rows).items()))
    print(
        f"{name}: n={len(rows):,} | Direct EM/F1={direct_em:.3f}/{direct_f1:.3f} "
        f"| Oracle EM/F1={oracle_em:.3f}/{oracle_f1:.3f} "
        f"| Gold-support EM/F1={float(np.mean([row['gold_support_em'] for row in rows])):.3f}/"
        f"{float(np.mean([row['gold_support_f1'] for row in rows])):.3f} "
        f"| Oracle gain EM/F1={oracle_em - direct_em:+.3f}/{oracle_f1 - direct_f1:+.3f} "
        f"| Graph-beneficial={graph_beneficial:.1%} | Oracle 행동={distribution}",
        flush=True,
    )
    _report_graph_diagnostics(rows, name)
    if all("sequential" in row for row in rows):
        _report_sequential_diagnostics(rows, name)
    return {
        "direct_em": direct_em,
        "direct_f1": direct_f1,
        "oracle_em": oracle_em,
        "oracle_f1": oracle_f1,
        "gold_support_em": float(np.mean([row["gold_support_em"] for row in rows])),
        "gold_support_f1": float(np.mean([row["gold_support_f1"] for row in rows])),
    }


def _report_graph_diagnostics(rows: list[dict], name: str) -> None:
    direct = [float(row["actions"][DIRECT_ACTION]["f1"]) for row in rows]
    action_summaries: list[str] = []
    for action, label in ACTION_LOG_NAMES.items():
        if not all(action in row["actions"] for row in rows):
            continue
        differences = [
            float(row["actions"][action]["f1"]) - direct_score
            for row, direct_score in zip(rows, direct, strict=True)
        ]
        action_summaries.append(
            f"{label} ΔF1={float(np.mean(differences)):+.3f} "
            f"(win={float(np.mean([gain > 1e-9 for gain in differences])):.0%}, "
            f"harm={float(np.mean([gain < -1e-9 for gain in differences])):.0%})"
        )

    beneficial = [row for row in rows if float(row["oracle_f1_gain"]) > 1e-9]
    evidence_aligned = [
        row
        for row in beneficial
        if float(row["actions"][row["oracle_action"]]["source_recall"])
        > float(row["actions"][DIRECT_ACTION]["source_recall"]) + 1e-9
    ]
    same_recall = len(beneficial) - len(evidence_aligned)
    print(
        f"{name} Graph 진단: "
        + " | ".join(action_summaries)
        + f" | Answer 개선 귀인: evidence-recall↑={len(evidence_aligned) / len(rows):.1%}, "
        f"same-recall={same_recall / len(rows):.1%}",
        flush=True,
    )


def _report_sequential_diagnostics(rows: list[dict], name: str) -> None:
    direct_f1 = float(np.mean([row["direct_f1"] for row in rows]))
    one_shot_f1 = float(np.mean([row["oracle_f1"] for row in rows]))
    sequential_f1 = float(np.mean([row["sequential_oracle_f1"] for row in rows]))
    trajectories = Counter(row["sequential_oracle_trajectory"] for row in rows)
    s0_actions = Counter(row["sequential"]["s0_oracle_action"] for row in rows)
    two_hop = float(
        np.mean(["expand_2hop" in row["sequential_oracle_trajectory"] for row in rows])
    )
    shared_oracle = all(
        np.isclose(float(row["oracle_f1"]), float(row["sequential_oracle_f1"]))
        and np.isclose(float(row["oracle_em"]), float(row["sequential_oracle_em"]))
        for row in rows
    )
    beneficial = [row for row in rows if row["sequential_oracle_f1_gain"] > 1e-9]
    evidence_aligned = 0
    for row in beneficial:
        trajectory = row["sequential_oracle_trajectory"]
        result = row["sequential"]["terminal_results"][trajectory]
        if float(result["source_recall"]) > float(row["actions"][DIRECT_ACTION]["source_recall"]) + 1e-9:
            evidence_aligned += 1
    print(
        f"{name} Sequential 상태 진단: Direct/One-shot Oracle/공유 Terminal Oracle F1="
        f"{direct_f1:.3f}/{one_shot_f1:.3f}/{sequential_f1:.3f} "
        f"| Oracle 일치={'yes' if shared_oracle else 'NO'} "
        f"| Oracle potential={sequential_f1 - direct_f1:+.3f} "
        f"| 2-hop 필요={two_hop:.1%} "
        f"| evidence-recall↑={evidence_aligned / len(rows):.1%} "
        f"| s0 행동={dict(sorted(s0_actions.items()))} "
        f"| trajectory={dict(sorted(trajectories.items()))}",
        flush=True,
    )


def _report_macro(rows: list[dict], datasets: Iterable[str]) -> None:
    groups = [[row for row in rows if row["dataset"] == dataset] for dataset in datasets]
    scores = [_report_values(group) for group in groups if group]
    if not scores:
        return
    sequential_text = ""
    if all("sequential_oracle_f1" in row for row in rows):
        sequential_text = (
            f" | Sequential Oracle EM/F1="
            f"{np.mean([score['sequential_oracle_em'] for score in scores]):.3f}/"
            f"{np.mean([score['sequential_oracle_f1'] for score in scores]):.3f}"
        )
    print(
        "Macro: "
        f"Direct EM/F1={np.mean([score['direct_em'] for score in scores]):.3f}/"
        f"{np.mean([score['direct_f1'] for score in scores]):.3f} "
        f"| Oracle EM/F1={np.mean([score['oracle_em'] for score in scores]):.3f}/"
        f"{np.mean([score['oracle_f1'] for score in scores]):.3f} "
        f"| Gold-support EM/F1={np.mean([score['gold_support_em'] for score in scores]):.3f}/"
        f"{np.mean([score['gold_support_f1'] for score in scores]):.3f}"
        f"{sequential_text}",
        flush=True,
    )


def _report_values(rows: list[dict]) -> dict[str, float]:
    values = {
        "direct_em": _average(rows, DIRECT_ACTION, "em"),
        "direct_f1": _average(rows, DIRECT_ACTION, "f1"),
        "oracle_em": float(np.mean([row["oracle_em"] for row in rows])),
        "oracle_f1": float(np.mean([row["oracle_f1"] for row in rows])),
        "gold_support_em": float(np.mean([row["gold_support_em"] for row in rows])),
        "gold_support_f1": float(np.mean([row["gold_support_f1"] for row in rows])),
    }
    if all("sequential_oracle_f1" in row for row in rows):
        values["sequential_oracle_em"] = float(
            np.mean([row["sequential_oracle_em"] for row in rows])
        )
        values["sequential_oracle_f1"] = float(
            np.mean([row["sequential_oracle_f1"] for row in rows])
        )
    return values


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
