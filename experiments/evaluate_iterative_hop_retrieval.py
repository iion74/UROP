"""Evaluate arbitrary-length STOP/CONTINUE search trajectories without a reader.

Queries can be sampled equally per gold hop for diagnosis only.  Gold hop is
never given to the retriever or stopping policy: every query produces a prefix
after each search step, and an oracle reports which prefix should have stopped.
``max_search_steps`` is a safety/cost cap rather than a predicted hop class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.evaluate_answer_oracle import _load_fixed_retriever
from experiments.train_multihop_policy import (
    SUPPORTED_DATASETS,
    MultiHopExample,
    _corpus_fingerprint,
    _dataset_seed,
    _duration,
    _iter_jsonl,
    evidence_metrics,
    load_processed_examples,
)
from src.iterative_retrieval import (
    BridgeSearchConfig,
    IterativeSearchConfig,
    compose_iterative_evidence,
    rank_iterative_chains,
)
from src.planned_retrieval import run_planned_retrieval
from src.qa_evaluation import ollama_model_identity
from src.query_rewriter import (
    FrozenOllamaQueryRewriter,
    GAP_PROMPT_VERSION,
    LEGACY_PROMPT_VERSION,
    QueryRewriteCache,
    QueryRewriterConfig,
)
from src.retriever import GRAPH_FORMAT_VERSION
from src.settings import settings


FORMAT_VERSION = 6
DIRECT_ACTION = "vector"


def main() -> None:
    args = _arguments()
    started = perf_counter()
    bridge_config = BridgeSearchConfig(
        seed_count=args.seed_count,
        graph_hops=1,
        candidates_per_seed=args.bridge_candidates_per_seed,
        anchor_text_characters=args.anchor_text_characters,
    )
    iterative_config = IterativeSearchConfig(
        max_evidence_hops=args.max_search_steps + 1,
        beam_width=args.beam_width,
        candidates_per_state=args.candidates_per_state,
        retrieval_candidates_per_state=args.retrieval_candidates_per_state,
        chain_text_characters=args.chain_text_characters,
    )
    bridge_config.validate()
    iterative_config.validate()
    actions = [DIRECT_ACTION]
    if args.enable_iterative:
        actions.extend(f"iterative_continue_{step}" for step in range(1, args.max_search_steps + 1))
    planner_config = None
    if args.enable_planner:
        try:
            resolved_model, model_digest = ollama_model_identity(
                args.ollama_host,
                args.planner_model,
                timeout_seconds=min(30, args.planner_timeout),
            )
        except RuntimeError as error:
            raise SystemExit(f"{error}\nOllama 실행 상태와 planner 모델을 확인하세요.") from error
        planner_config = QueryRewriterConfig(
            host=args.ollama_host,
            model=resolved_model,
            model_digest=model_digest,
            max_tokens=args.planner_max_tokens,
            timeout_seconds=args.planner_timeout,
            seed=args.planner_seed,
            keep_alive=args.ollama_keep_alive,
            prompt_version=args.planner_prompt_version,
        )
        actions.extend(f"planned_continue_{step}" for step in range(1, args.max_search_steps + 1))
    actions = tuple(actions)
    _log("상태 기반 STOP/CONTINUE retrieval 실험을 시작합니다. gold hop은 표본/평가에만 사용됩니다.")
    _log(
        f"설정: report_hops={args.report_hops}, queries_by_hop={args.query_counts}, "
        f"max_evidence_hops={args.max_evidence_hops}, "
        f"max_search_steps={args.max_search_steps}(안전 상한), top_k={args.top_k}, "
        f"beam={args.beam_width}, candidates/state={args.candidates_per_state}"
    )

    shared_embedding_model = None
    shared_reranker = None
    all_rows: list[dict] = []
    with ExitStack() as stack:
        planner = None
        if planner_config is not None:
            cache = stack.enter_context(QueryRewriteCache(Path(args.planner_cache)))
            planner_session = requests.Session()
            stack.callback(planner_session.close)
            planner = FrozenOllamaQueryRewriter(planner_config, cache, session=planner_session)
            _log(
                f"고정 query planner: model={planner_config.model}, digest={planner_config.model_digest}, "
                f"prompt={planner_config.prompt_version}"
            )
        for dataset in args.datasets:
            dataset_rows, shared_embedding_model, shared_reranker = _run_dataset(
                args,
                dataset,
                bridge_config,
                iterative_config,
                actions,
                planner,
                planner_config,
                shared_embedding_model,
                shared_reranker,
            )
            all_rows.extend(dataset_rows)
    if len(args.datasets) > 1:
        _report(all_rows, "Macro/Micro(equal-hop cohort)", args.report_hops, actions)
    _log(f"STOP/CONTINUE trajectory retrieval 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _run_dataset(
    args,
    dataset: str,
    bridge_config: BridgeSearchConfig,
    iterative_config: IterativeSearchConfig,
    actions: tuple[str, ...],
    planner,
    planner_config,
    shared_embedding_model,
    shared_reranker,
):
    input_path = Path(args.processed_root) / dataset / f"{args.split}.jsonl"
    corpus_examples = load_processed_examples(
        input_path,
        args.corpus_examples_per_dataset,
        _dataset_seed(args.corpus_seed, dataset),
    )
    examples = _select_per_hop(
        corpus_examples,
        args.report_hops,
        args.query_counts,
        _dataset_seed(args.query_seed, dataset),
    )
    distribution = dict(sorted(Counter(example.hop_count for example in examples).items()))
    _log(f"{dataset}: 균형 query {len(examples):,}개 · hop {distribution}")
    corpus_fingerprint = _corpus_fingerprint(corpus_examples)
    retriever = _load_fixed_retriever(
        Path(args.index_root) / args.split / dataset,
        corpus_fingerprint,
        embedding_model=shared_embedding_model,
        reranker=shared_reranker,
    )
    shared_embedding_model = retriever.model or shared_embedding_model
    shared_reranker = retriever.reranker or shared_reranker
    output_path = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
    manifest_config = {
        "format_version": FORMAT_VERSION,
        "experiment": "structured_gap_evidence_survival_retrieval_v6",
        "dataset": dataset,
        "split": args.split,
        "corpus_fingerprint": corpus_fingerprint,
        "corpus_examples_per_dataset": args.corpus_examples_per_dataset,
        "corpus_seed": args.corpus_seed,
        "query_seed": args.query_seed,
        "report_hops": list(args.report_hops),
        "max_evidence_hops": args.max_evidence_hops,
        "max_search_steps": args.max_search_steps,
        "queries_by_hop": {str(hop): count for hop, count in args.query_counts.items()},
        "top_k": args.top_k,
        "initial_candidates": args.initial_candidates,
        "actions": list(actions),
        "enable_iterative": args.enable_iterative,
        "graph_format_version": GRAPH_FORMAT_VERSION,
        "bridge_config": asdict(bridge_config),
        "iterative_config": asdict(iterative_config),
        "planner_config": asdict(planner_config) if planner_config is not None else None,
        "planner_search_candidates": args.planner_search_candidates,
        "planner_context_characters": args.planner_context_characters,
        "planner_anchor_slots": args.planner_anchor_slots,
        "planner_reader_source_cap": args.planner_reader_source_cap,
        "planner_additions_per_step": args.planner_additions_per_step,
        "planner_replace_when_full": args.planner_replace_when_full,
    }
    rows = _evaluate_dataset(
        retriever,
        examples,
        output_path,
        manifest_config,
        bridge_config=bridge_config,
        iterative_config=iterative_config,
        actions=actions,
        top_k=args.top_k,
        initial_candidates=args.initial_candidates,
        log_every=args.log_every,
        overwrite=args.overwrite,
        run_iterative=args.enable_iterative,
        planner=planner,
        planner_search_candidates=args.planner_search_candidates,
        planner_context_characters=args.planner_context_characters,
        planner_anchor_slots=args.planner_anchor_slots,
        planner_reader_source_cap=args.planner_reader_source_cap,
        planner_additions_per_step=args.planner_additions_per_step,
        planner_replace_when_full=args.planner_replace_when_full,
    )
    _report(rows, dataset, args.report_hops, actions)
    return rows, retriever.model or shared_embedding_model, retriever.reranker or shared_reranker


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="임의 깊이 STOP/CONTINUE 순차 retrieval 실험")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=["musique"])
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--output-root", default="data/benchmarks/multihop/sequential_search_v6_gap_memory")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument(
        "--report-hops",
        "--hops",
        dest="report_hops",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="균형 표본/사후 보고용 gold hop. 검색기나 정책 입력에는 전달되지 않습니다.",
    )
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument(
        "--max-evidence-hops",
        type=int,
        help="최대 evidence hop. 기본 4이며 초기 검색 뒤 최대 3회 CONTINUE합니다.",
    )
    depth_group.add_argument(
        "--max-search-steps",
        type=int,
        help="초기 검색 이후 CONTINUE 실행 상한. 4-hop은 3 step입니다.",
    )
    parser.add_argument("--queries-per-hop", type=int, default=20)
    parser.add_argument(
        "--queries-by-hop",
        nargs="+",
        metavar="HOP:COUNT",
        help="hop별 서로 다른 표본 수. 예: 2:400 3:300 4:300",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--initial-candidates", type=int, default=15)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--bridge-candidates-per-seed", type=int, default=12)
    parser.add_argument("--anchor-text-characters", type=int, default=700)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--candidates-per-state", type=int, default=8)
    parser.add_argument("--retrieval-candidates-per-state", type=int, default=8)
    parser.add_argument("--chain-text-characters", type=int, default=1200)
    executor_group = parser.add_mutually_exclusive_group()
    executor_group.add_argument("--enable-iterative", action="store_true")
    executor_group.add_argument(
        "--skip-iterative",
        dest="enable_iterative",
        action="store_false",
        help="이전 명령 호환용. iterative executor는 기본적으로 비활성입니다.",
    )
    parser.set_defaults(enable_iterative=False)
    parser.add_argument("--enable-planner", action="store_true")
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--planner-model", default=settings.ollama_model)
    parser.add_argument("--planner-cache", default="data/benchmarks/multihop/query_rewriter_cache_v2.sqlite3")
    parser.add_argument("--planner-max-tokens", type=int, default=80)
    parser.add_argument("--planner-timeout", type=int, default=300)
    parser.add_argument("--planner-seed", type=int, default=42)
    parser.add_argument(
        "--planner-prompt-version",
        choices=(LEGACY_PROMPT_VERSION, GAP_PROMPT_VERSION),
        default=LEGACY_PROMPT_VERSION,
    )
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--planner-search-candidates", type=int, default=8)
    parser.add_argument("--planner-context-characters", type=int, default=4000)
    parser.add_argument(
        "--planner-anchor-slots",
        type=int,
        default=2,
        help="후속 검색이 교체하지 못하도록 보존할 최초 검색 근거 수",
    )
    parser.add_argument(
        "--planner-reader-source-cap",
        type=int,
        default=5,
        help="교체 대신 누적 보존할 reader 근거 수 상한",
    )
    parser.add_argument(
        "--planner-additions-per-step",
        type=int,
        default=1,
        help="한 번의 재작성 검색에서 reader에 추가할 서로 다른 자료 수",
    )
    parser.add_argument(
        "--planner-freeze-when-full",
        dest="planner_replace_when_full",
        action="store_false",
        help="reader 근거 상한에 도달하면 기존 근거를 교체하지 않습니다.",
    )
    parser.set_defaults(planner_replace_when_full=True)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    args.report_hops = tuple(sorted(set(args.report_hops)))
    if args.queries_by_hop:
        try:
            query_counts = {
                int(item.split(":", 1)[0]): int(item.split(":", 1)[1])
                for item in args.queries_by_hop
            }
        except (ValueError, IndexError) as error:
            parser.error(f"--queries-by-hop 형식은 HOP:COUNT입니다: {error}")
        if set(query_counts) != set(args.report_hops):
            parser.error("--queries-by-hop의 hop은 --report-hops와 정확히 같아야 합니다.")
        args.query_counts = query_counts
    else:
        args.query_counts = {hop: args.queries_per_hop for hop in args.report_hops}
    if args.max_evidence_hops is None and args.max_search_steps is None:
        args.max_evidence_hops = 4
        args.max_search_steps = 3
    elif args.max_evidence_hops is not None:
        args.max_search_steps = args.max_evidence_hops - 1
    else:
        args.max_evidence_hops = args.max_search_steps + 1
    if not args.report_hops or min(args.report_hops) < 1:
        parser.error("보고할 hop은 1 이상이어야 합니다.")
    if args.max_evidence_hops < 2 or args.max_search_steps < 1:
        parser.error("max-evidence-hops는 2 이상이어야 합니다.")
    if args.corpus_examples_per_dataset < 1 or any(count < 1 for count in args.query_counts.values()):
        parser.error("corpus와 hop별 query 수는 1 이상이어야 합니다.")
    if args.initial_candidates < args.top_k:
        parser.error("initial candidates는 top-k 이상이어야 합니다.")
    if args.log_every < 1:
        parser.error("log 간격은 1 이상이어야 합니다.")
    if not args.enable_iterative and not args.enable_planner:
        parser.error("--enable-planner 또는 --enable-iterative 중 하나를 지정하세요.")
    if (
        args.planner_max_tokens < 1
        or args.planner_timeout < 1
        or args.planner_search_candidates < 1
        or args.planner_context_characters < 1
        or args.planner_additions_per_step < 1
        or not 0 <= args.planner_anchor_slots <= args.top_k
        or args.planner_reader_source_cap < args.top_k
    ):
        parser.error("planner token/timeout/candidate/context 값은 1 이상이어야 합니다.")
    return args


def _select_per_hop(
    examples: Iterable[MultiHopExample],
    hops: tuple[int, ...],
    per_hop: int | Mapping[int, int],
    seed: int,
) -> list[MultiHopExample]:
    selected: list[MultiHopExample] = []
    examples = tuple(examples)
    for hop in hops:
        requested = int(per_hop[hop]) if isinstance(per_hop, Mapping) else int(per_hop)
        available = [example for example in examples if example.hop_count == hop]
        if len(available) < requested:
            raise SystemExit(
                f"hop={hop}: 요청 {requested:,}개보다 corpus cohort의 문항 {len(available):,}개가 적습니다."
            )
        ranked = sorted(
            available,
            key=lambda example: (
                hashlib.sha256(f"{seed}\0hop={hop}\0{example.policy_id}".encode("utf-8")).digest(),
                example.policy_id,
            ),
        )
        selected.extend(ranked[:requested])
    return sorted(selected, key=lambda example: example.policy_id)


def _evaluate_dataset(
    retriever,
    examples: list[MultiHopExample],
    output_path: Path,
    manifest_config: Mapping,
    *,
    bridge_config: BridgeSearchConfig,
    iterative_config: IterativeSearchConfig,
    actions: tuple[str, ...],
    top_k: int,
    initial_candidates: int,
    log_every: int,
    overwrite: bool,
    run_iterative: bool = True,
    planner=None,
    planner_search_candidates: int = 8,
    planner_context_characters: int = 4000,
    planner_anchor_slots: int = 2,
    planner_reader_source_cap: int = 5,
    planner_additions_per_step: int = 1,
    planner_replace_when_full: bool = True,
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
                f"기존 iterative-hop 설정과 현재 설정이 다릅니다: {output_path}\n"
                "다른 --output-root를 쓰거나 --overwrite를 지정하세요."
            )
    expected_ids = {example.policy_id for example in examples}
    rows_by_id = {row_id: row for row_id, row in cached.items() if row_id in expected_ids}
    if len(rows_by_id) == len(examples):
        _log(f"{manifest_config['dataset']}: {len(rows_by_id):,}개 iterative-hop 행 전체 재사용")
        return [rows_by_id[example.policy_id] for example in examples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite or not output_path.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    _write_manifest(manifest_path, manifest_config, "in_progress", len(rows_by_id), len(examples))
    started = perf_counter()
    newly_completed = 0
    with output_path.open(mode, encoding="utf-8") as destination:
        for example in examples:
            if example.policy_id in rows_by_id:
                continue
            initial = retriever.search(example.question, top_k=initial_candidates)
            result = None
            if run_iterative:
                result = rank_iterative_chains(
                    retriever,
                    example.question,
                    initial,
                    bridge_config,
                    iterative_config,
                )
            sources = {DIRECT_ACTION: initial[:top_k]}
            if result is not None:
                for step in range(1, int(manifest_config["max_search_steps"]) + 1):
                    sources[f"iterative_continue_{step}"] = compose_iterative_evidence(
                        initial,
                        result.chains_by_hop,
                        evidence_hops=step + 1,
                        top_k=top_k,
                    )
            planned = None
            if planner is not None:
                planned = run_planned_retrieval(
                    retriever,
                    planner,
                    example.question,
                    initial,
                    max_search_steps=int(manifest_config["max_search_steps"]),
                    top_k=top_k,
                    search_candidates=planner_search_candidates,
                    context_characters=planner_context_characters,
                    anchor_slots=planner_anchor_slots,
                    reader_source_cap=planner_reader_source_cap,
                    additions_per_step=planner_additions_per_step,
                    replace_when_full=planner_replace_when_full,
                )
                for step in range(1, int(manifest_config["max_search_steps"]) + 1):
                    planned_sources = planned.sources_by_step.get(step, tuple(initial[:top_k]))
                    sources[f"planned_continue_{step}"] = list(planned_sources)
            action_results = {
                action: _score_retrieval(action_sources, example.supporting_groups)
                for action, action_sources in sources.items()
            }
            oracle_action = max(
                actions,
                key=lambda action: (
                    float(action_results[action]["all_support_recall"]),
                    float(action_results[action]["source_recall"]),
                    -actions.index(action),
                ),
            )
            row = {
                "format_version": FORMAT_VERSION,
                "id": example.policy_id,
                "dataset": example.dataset,
                "question": example.question,
                "gold_hop": example.hop_count,
                "reasoning_type": example.reasoning_type,
                "actions": action_results,
                "oracle_action": oracle_action,
                "oracle_stop_step": _action_stop_step(oracle_action),
                "planned_stop_labels": _planned_stop_labels(
                    action_results,
                    int(manifest_config["max_search_steps"]),
                ),
                "iterative_audit": ({
                    "evaluated_pairs_by_hop": {
                        str(hop): int(count) for hop, count in result.evaluated_pairs_by_hop.items()
                    },
                    "chain_counts_by_hop": {
                        str(hop): len(chains) for hop, chains in result.chains_by_hop.items()
                    },
                    "top_chain_scores": {
                        str(hop): (chains[0].score if chains else 0.0)
                        for hop, chains in result.chains_by_hop.items()
                    },
                    "top_chain_content_ids": {
                        str(hop): [chunk.source for chunk in chains[0].chunks] if chains else []
                        for hop, chains in result.chains_by_hop.items()
                    },
                    "beam_chain_content_ids": {
                        str(hop): [
                            [chunk.source for chunk in chain.chunks]
                            for chain in chains
                        ]
                        for hop, chains in result.chains_by_hop.items()
                    },
                } if result is not None else {}),
                "planner_audit": (
                    [
                        {
                            "search_step": step.search_step,
                            "decision": "CONTINUE" if step.decision.continue_search else "STOP",
                            "policy_state": {
                                "completed_steps": step.state.search_step,
                                "reader_source_count": len(step.state.reader_sources),
                                "memory_size": len(step.state.memory),
                                "previous_query_count": len(step.state.previous_queries),
                                "last_added_score": step.state.last_added_score,
                                "consecutive_no_new": step.state.consecutive_no_new,
                                "evidence_sufficient": step.state.evidence_sufficient,
                                "missing_information": step.state.missing_information,
                            },
                            "query": step.rewrite.query,
                            "evidence_sufficient": step.rewrite.sufficient,
                            "missing_information": step.rewrite.missing_information,
                            "cache_hit": step.rewrite.cache_hit,
                            "latency_seconds": step.rewrite.latency_seconds,
                            "added_content_id": step.added_chunk.source if step.added_chunk else None,
                            "added_title": step.added_chunk.title if step.added_chunk else None,
                            "added_score": step.added_score,
                            "added_content_ids": [chunk.source for chunk in step.added_chunks],
                            "added_titles": [chunk.title for chunk in step.added_chunks],
                            "added_scores": list(step.added_scores),
                        }
                        for step in planned.steps
                    ]
                    if planned is not None
                    else []
                ),
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
                parts = "/".join(f"{_mean(current, action, 'all_support_recall'):.3f}" for action in actions)
                _log(
                    f"{example.dataset} trajectory {completed}/{len(examples)} ({completed / len(examples):.1%}) "
                    f"| ETA {_duration(eta)} | All-Support {'/'.join(actions)}={parts}"
                )
                _write_manifest(manifest_path, manifest_config, "in_progress", completed, len(examples))
    _write_manifest(manifest_path, manifest_config, "ready", len(rows_by_id), len(examples))
    return [rows_by_id[example.policy_id] for example in examples]


def _score_retrieval(sources, supporting_groups) -> dict:
    retrieved = {chunk.source for chunk, _ in sources}
    recall, all_support = evidence_metrics(retrieved, supporting_groups)
    return {
        "source_recall": recall,
        "all_support_recall": all_support,
        "evidence_content_ids": [chunk.source for chunk, _ in sources],
        "evidence": [
            {
                "chunk_id": chunk.chunk_id,
                "content_id": chunk.source,
                "title": chunk.title,
                "score": float(score),
            }
            for chunk, score in sources
        ],
    }


def _report(rows: list[dict], name: str, hops: tuple[int, ...], actions: tuple[str, ...]) -> None:
    for gold_hop in (*hops, "Macro-hop"):
        selected = rows if gold_hop == "Macro-hop" else [row for row in rows if row["gold_hop"] == gold_hop]
        if not selected:
            continue
        values = " | ".join(
            f"{action} Recall/All={_mean(selected, action, 'source_recall'):.3f}/"
            f"{_mean(selected, action, 'all_support_recall'):.3f}"
            for action in actions
        )
        oracle_recall = float(
            np.mean([row["actions"][row["oracle_action"]]["source_recall"] for row in selected])
        )
        oracle_all = float(
            np.mean([row["actions"][row["oracle_action"]]["all_support_recall"] for row in selected])
        )
        distribution = dict(sorted(Counter(row["oracle_action"] for row in selected).items()))
        _log(
            f"{name} hop={gold_hop}: n={len(selected)} | {values} | "
            f"Stop-step Oracle Recall/All={oracle_recall:.3f}/{oracle_all:.3f} · 행동={distribution}"
        )


def _mean(rows: list[dict], action: str, metric: str) -> float:
    return float(np.mean([float(row["actions"][action][metric]) for row in rows])) if rows else 0.0


def _action_stop_step(action: str) -> int:
    if action == DIRECT_ACTION:
        return 0
    try:
        return int(action.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return 0


def _planned_stop_labels(action_results: Mapping[str, Mapping], max_search_steps: int) -> list[dict]:
    """Create offline STOP/CONTINUE labels without exposing gold at inference.

    These retrieval labels are diagnostic.  The final policy uses the same
    prefix comparison with frozen-reader Answer F1 as utility.
    """

    actions = [DIRECT_ACTION, *(f"planned_continue_{step}" for step in range(1, max_search_steps + 1))]
    if any(action not in action_results for action in actions):
        return []
    labels = []
    for state_step in range(0, max_search_steps):
        current_action = actions[state_step]
        future_actions = actions[state_step + 1 :]
        best_future = max(future_actions, key=lambda action: _retrieval_utility(action_results[action]))
        current_utility = _retrieval_utility(action_results[current_action])
        future_utility = _retrieval_utility(action_results[best_future])
        should_continue = future_utility > current_utility
        labels.append(
            {
                "state_step": state_step,
                "label": "CONTINUE" if should_continue else "STOP",
                "current_action": current_action,
                "best_future_action": best_future,
                "all_support_gain": float(action_results[best_future]["all_support_recall"])
                - float(action_results[current_action]["all_support_recall"]),
                "source_recall_gain": float(action_results[best_future]["source_recall"])
                - float(action_results[current_action]["source_recall"]),
            }
        )
    return labels


def _retrieval_utility(result: Mapping) -> tuple[float, float]:
    return (float(result["all_support_recall"]), float(result["source_recall"]))


def _manifest_matches(path: Path, expected: Mapping) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        ignored = {"queries_per_hop", "queries_by_hop", "status", "completed_rows", "requested_rows"}
        # These options were added after the evidence-survival v6 trajectories
        # had already been generated.  Missing values in an older manifest are
        # exactly the original one-addition/replacement behavior, so they are
        # safe to reuse when expanding the deterministic query cohort.
        legacy_defaults = {
            "planner_additions_per_step": 1,
            "planner_replace_when_full": True,
        }
        # v5 used the same executor defaults as a five-source v6 run, before
        # the evidence-survival controls were written explicitly to manifests.
        # Permit that one exact legacy configuration so a deterministic 1,000
        # row cohort can be extended to the containing 2,000-row cohort.  The
        # strict guards prevent accidental reuse for the new 5->8 growth run.
        legacy_v5 = (
            saved.get("format_version") == 5
            and saved.get("experiment") == "state_based_stop_continue_retrieval_v5"
            and expected.get("format_version") == 6
            and expected.get("experiment") == "structured_gap_evidence_survival_retrieval_v6"
            and saved.get("top_k") == 5
            and expected.get("top_k") == 5
            and expected.get("planner_anchor_slots") == 2
            and expected.get("planner_reader_source_cap") == 5
            and expected.get("planner_additions_per_step") == 1
            and expected.get("planner_replace_when_full") is True
        )
        comparable_saved = dict(saved)
        if legacy_v5:
            comparable_saved.update({
                "format_version": 6,
                "experiment": "structured_gap_evidence_survival_retrieval_v6",
                "planner_anchor_slots": 2,
                "planner_reader_source_cap": 5,
            })
        return all(
            comparable_saved.get(key, legacy_defaults.get(key)) == value
            for key, value in expected.items()
            if key not in ignored
        )
    except (OSError, ValueError, TypeError):
        return False


def _write_manifest(path: Path, config: Mapping, status: str, completed: int, requested: int) -> None:
    payload = dict(config)
    payload.update({"status": status, "completed_rows": completed, "requested_rows": requested})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
