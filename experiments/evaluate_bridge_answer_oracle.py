"""Answer-F1 pilot for evidence-conditioned bridge retrieval actions.

The command reuses the completed legacy Answer Oracle rows for Direct and the
four old Graph actions.  It calls the frozen reader only for ``bridge_path_1``
and ``bridge_path_2``, then reports how much the enlarged action space raises
the old Answer Oracle ceiling.  No policy is trained here.

Run the cheaper ``evaluate_bridge_retrieval`` screen first.  Only promising
bridge settings should reach this answer-level experiment.
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

from experiments.evaluate_answer_oracle import _load_fixed_retriever, _score_sources
from experiments.train_multihop_policy import (
    SUPPORTED_DATASETS,
    _corpus_fingerprint,
    _dataset_seed,
    _duration,
    _iter_jsonl,
    load_processed_examples,
    select_query_examples,
)
from src.iterative_retrieval import (
    BridgeSearchConfig,
    compose_bridge_evidence,
    rank_bridge_paths,
)
from src.qa_evaluation import (
    PROMPT_VERSION,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    ollama_model_identity,
)
from src.retriever import GRAPH_FORMAT_VERSION
from src.settings import settings


FORMAT_VERSION = 1
NEW_ACTIONS = ("bridge_path_1", "bridge_path_2")


def main() -> None:
    args = _arguments()
    started = perf_counter()
    bridge_config = BridgeSearchConfig(
        seed_count=args.seed_count,
        graph_hops=args.graph_hops,
        candidates_per_seed=args.candidates_per_seed,
        anchor_text_characters=args.anchor_text_characters,
    )
    bridge_config.validate()

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
    _log("증거 조건부 Bridge Answer Oracle pilot을 시작합니다. 기존 행동은 cache를 재사용합니다.")
    _log(
        f"Reader 고정: model={resolved_model}, digest={model_digest or '미제공'}, "
        f"temperature=0, prompt={PROMPT_VERSION}"
    )
    _log(
        f"새 행동: bridge_path_1,bridge_path_2 · seed_count={bridge_config.seed_count} · "
        f"graph_hops={bridge_config.graph_hops} · candidates/seed={bridge_config.candidates_per_seed}"
    )

    all_rows: list[dict] = []
    shared_embedding_model = None
    shared_reranker = None
    with AnswerCache(Path(args.cache_path)) as cache:
        reader = FrozenOllamaReader(reader_config, cache, session=session)
        for dataset in args.datasets:
            input_path = Path(args.processed_root) / dataset / f"{args.split}.jsonl"
            corpus_examples = load_processed_examples(
                input_path,
                args.corpus_examples_per_dataset,
                _dataset_seed(args.corpus_seed, dataset),
            )
            examples = select_query_examples(
                corpus_examples,
                args.max_queries_per_dataset,
                _dataset_seed(args.query_seed, dataset),
            )
            if not examples:
                raise SystemExit(f"유효한 query가 없습니다: {input_path}")
            expected_ids = {example.policy_id for example in examples}
            base_path = Path(args.base_oracle_root) / f"{dataset}_{args.split}.jsonl"
            base_rows = {
                str(row["id"]): row
                for row in _iter_jsonl(base_path)
                if str(row.get("id")) in expected_ids
            }
            missing = expected_ids - set(base_rows)
            if missing:
                raise SystemExit(
                    f"{dataset}: 기존 Answer Oracle에 필요한 문항이 없습니다: {next(iter(missing))}\n"
                    "먼저 evaluate_answer_oracle을 같은 corpus/query seed로 실행하세요."
                )

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
                "experiment": "evidence_conditioned_bridge_answer_oracle_v1",
                "dataset": dataset,
                "split": args.split,
                "base_oracle": str(base_path),
                "base_oracle_fingerprint": _file_fingerprint(base_path),
                "corpus_fingerprint": corpus_fingerprint,
                "corpus_seed": args.corpus_seed,
                "query_seed": args.query_seed,
                "corpus_examples_per_dataset": args.corpus_examples_per_dataset,
                "max_queries_per_dataset": args.max_queries_per_dataset,
                "top_k": args.top_k,
                "initial_candidates": args.initial_candidates,
                "context_character_budget": args.context_character_budget,
                "new_actions": list(NEW_ACTIONS),
                "graph_format_version": GRAPH_FORMAT_VERSION,
                "bridge_config": asdict(bridge_config),
                "reader": {**asdict(reader_config), "prompt_version": PROMPT_VERSION},
            }
            rows = _evaluate_dataset(
                retriever,
                reader,
                examples,
                base_rows,
                output_path,
                manifest_config,
                config=bridge_config,
                top_k=args.top_k,
                initial_candidates=args.initial_candidates,
                context_character_budget=args.context_character_budget,
                log_every=args.log_every,
                overwrite=args.overwrite,
            )
            all_rows.extend(rows)
            _report(rows, dataset)

    _report_macro(all_rows, args.datasets)
    _log(f"Bridge Answer Oracle pilot 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="중간 증거 기반 Bridge 행동의 Answer EM/F1 평가")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--base-oracle-root", default="data/benchmarks/multihop/answer_oracle")
    parser.add_argument("--output-root", default="data/benchmarks/multihop/bridge_answer_oracle_v1")
    parser.add_argument("--cache-path", default="data/benchmarks/multihop/answer_cache.sqlite3")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--max-queries-per-dataset", type=int, default=100)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--initial-candidates", type=int, default=15)
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--graph-hops", type=int, default=1)
    parser.add_argument("--candidates-per-seed", type=int, default=12)
    parser.add_argument("--anchor-text-characters", type=int, default=700)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--reader-model", default=settings.ollama_model)
    parser.add_argument("--reader-max-tokens", type=int, default=64)
    parser.add_argument("--reader-timeout", type=int, default=300)
    parser.add_argument("--reader-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
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
    if args.context_character_budget < 1 or args.reader_max_tokens < 1 or args.reader_timeout < 1:
        parser.error("context/token/timeout은 1 이상이어야 합니다.")
    return args


def _evaluate_dataset(
    retriever,
    reader,
    examples,
    base_rows: Mapping[str, dict],
    output_path: Path,
    manifest_config: Mapping,
    *,
    config: BridgeSearchConfig,
    top_k: int,
    initial_candidates: int,
    context_character_budget: int,
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
                f"기존 Bridge Answer 설정과 현재 설정이 다릅니다: {output_path}\n"
                "다른 --output-root를 쓰거나 --overwrite를 지정하세요."
            )
    expected_ids = {example.policy_id for example in examples}
    rows_by_id = {row_id: row for row_id, row in cached.items() if row_id in expected_ids}
    if len(rows_by_id) == len(examples):
        _log(f"{manifest_config['dataset']}: Bridge Answer {len(rows_by_id):,}개 전체 재사용")
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
            base = base_rows[example.policy_id]
            if str(base.get("question")) != example.question:
                raise SystemExit(f"기존 Oracle 질문이 전처리 데이터와 다릅니다: {example.policy_id}")

            initial_started = perf_counter()
            initial = retriever.search(example.question, top_k=initial_candidates)
            initial_latency = perf_counter() - initial_started
            bridge_started = perf_counter()
            bridge = rank_bridge_paths(retriever, example.question, initial, config)
            bridge_latency = perf_counter() - bridge_started
            new_sources = {
                "bridge_path_1": compose_bridge_evidence(initial, bridge.paths, top_k=top_k, max_paths=1),
                "bridge_path_2": compose_bridge_evidence(initial, bridge.paths, top_k=top_k, max_paths=2),
            }
            new_results = {
                action: _score_sources(
                    reader,
                    example,
                    sources,
                    extra_candidates=bridge.distinct_candidates,
                    initial_retrieval_latency=initial_latency,
                    action_retrieval_latency=bridge_latency,
                    context_character_budget=context_character_budget,
                )
                for action, sources in new_sources.items()
            }
            combined = {**base["actions"], **new_results}
            order = tuple(base["actions"]) + NEW_ACTIONS
            expanded_action = max(
                order,
                key=lambda action: (
                    float(combined[action]["f1"]),
                    float(combined[action]["em"]),
                    -int(combined[action].get("extra_candidates", 0)),
                    -order.index(action),
                ),
            )
            expanded = combined[expanded_action]
            row = {
                "format_version": FORMAT_VERSION,
                "id": example.policy_id,
                "dataset": example.dataset,
                "question": example.question,
                "hop_count": example.hop_count,
                "reasoning_type": example.reasoning_type,
                "direct_em": float(base["direct_em"]),
                "direct_f1": float(base["direct_f1"]),
                "old_oracle_action": str(base["oracle_action"]),
                "old_oracle_em": float(base["oracle_em"]),
                "old_oracle_f1": float(base["oracle_f1"]),
                "new_actions": {
                    action: _compact_action(result) for action, result in new_results.items()
                },
                "expanded_oracle_action": expanded_action,
                "expanded_oracle_em": float(expanded["em"]),
                "expanded_oracle_f1": float(expanded["f1"]),
                "expanded_oracle_em_gain": float(expanded["em"]) - float(base["direct_em"]),
                "expanded_oracle_f1_gain": float(expanded["f1"]) - float(base["direct_f1"]),
                "new_oracle_f1_gain": float(expanded["f1"]) - float(base["oracle_f1"]),
                "bridge_audit": {
                    "evaluated_pairs": bridge.evaluated_pairs,
                    "distinct_candidates": bridge.distinct_candidates,
                    "top_path_score": bridge.paths[0].score if bridge.paths else 0.0,
                    "top_path_margin": (
                        bridge.paths[0].score - bridge.paths[1].score if len(bridge.paths) > 1 else 0.0
                    ),
                    "top_path_title_mentioned": bridge.paths[0].title_mentioned if bridge.paths else False,
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
                    f"{example.dataset} Answer {completed:,}/{len(examples):,} ({completed / len(examples):.1%}) "
                    f"| ETA {_duration(eta)} | Direct/OldOracle/NewOracle F1="
                    f"{_mean_key(current, 'direct_f1'):.3f}/"
                    f"{_mean_key(current, 'old_oracle_f1'):.3f}/"
                    f"{_mean_key(current, 'expanded_oracle_f1'):.3f} "
                    f"| 새 Oracle gain={_mean_key(current, 'new_oracle_f1_gain'):+.3f}"
                )
                _write_manifest(manifest_path, manifest_config, "in_progress", completed)
    _write_manifest(manifest_path, manifest_config, "ready", len(rows_by_id))
    return [rows_by_id[example.policy_id] for example in examples]


def _compact_action(result: Mapping) -> dict:
    keep = (
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
    return {
        **{key: result[key] for key in keep if key in result},
        "evidence_content_ids": [str(item["content_id"]) for item in result.get("evidence", ())],
    }


def _report(rows: list[dict], name: str) -> None:
    distribution = dict(sorted(Counter(str(row["expanded_oracle_action"]) for row in rows).items()))
    direct = _mean_key(rows, "direct_f1")
    old = _mean_key(rows, "old_oracle_f1")
    new = _mean_key(rows, "expanded_oracle_f1")
    bridge_1 = _mean_action(rows, "bridge_path_1", "f1")
    bridge_2 = _mean_action(rows, "bridge_path_2", "f1")
    _log(
        f"{name}: n={len(rows):,} | Direct/Bridge1/Bridge2/OldOracle/NewOracle F1="
        f"{direct:.3f}/{bridge_1:.3f}/{bridge_2:.3f}/{old:.3f}/{new:.3f} "
        f"| New−Old Oracle={new - old:+.3f} | New−Direct={new - direct:+.3f} "
        f"| 행동={distribution}"
    )


def _report_macro(rows: list[dict], datasets: Iterable[str]) -> None:
    groups = [[row for row in rows if row["dataset"] == dataset] for dataset in datasets]
    groups = [group for group in groups if group]
    if not groups:
        return
    direct = float(np.mean([_mean_key(group, "direct_f1") for group in groups]))
    old = float(np.mean([_mean_key(group, "old_oracle_f1") for group in groups]))
    new = float(np.mean([_mean_key(group, "expanded_oracle_f1") for group in groups]))
    bridge_1 = float(np.mean([_mean_action(group, "bridge_path_1", "f1") for group in groups]))
    bridge_2 = float(np.mean([_mean_action(group, "bridge_path_2", "f1") for group in groups]))
    _log(
        f"Macro: Direct/Bridge1/Bridge2/OldOracle/NewOracle F1="
        f"{direct:.3f}/{bridge_1:.3f}/{bridge_2:.3f}/{old:.3f}/{new:.3f} "
        f"| New−Old Oracle={new - old:+.3f} | New−Direct={new - direct:+.3f}"
    )


def _mean_key(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _mean_action(rows: list[dict], action: str, metric: str) -> float:
    return float(np.mean([float(row["new_actions"][action][metric]) for row in rows])) if rows else 0.0


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_matches(path: Path, expected: Mapping) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        # Query cohorts are nested by stable hash, so completed pilot rows can
        # be reused when the requested cohort grows from 20 to 100/1,000.
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
