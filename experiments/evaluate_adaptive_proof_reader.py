"""Evaluate adaptive proof-completeness retrieval with the frozen reader."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests

from experiments.evaluate_answer_oracle import _load_fixed_retriever
from experiments.train_counterfactual_proof_policy import _load_split
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _corpus_fingerprint, _duration, _iter_jsonl
from src.qa_evaluation import (
    PROMPT_VERSION,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    answer_score,
    evidence_context,
    ollama_model_identity,
)
from src.settings import settings


FORMAT_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive proof reader evaluation")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument(
        "--trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument(
        "--policy-predictions",
        default="data/models/adaptive_proof_completion_policy.report.predictions.jsonl",
    )
    parser.add_argument(
        "--direct-answer-root",
        default="data/benchmarks/multihop/stop_answer_oracle_v7_evidence_survival_validation_2000",
    )
    parser.add_argument(
        "--fixed-proof-root",
        default="data/benchmarks/multihop/causal_evidence_reader/factorized_graph_repair_validation_2000",
    )
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-path", default="data/benchmarks/multihop/answer_cache.sqlite3")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-rows-per-dataset", type=int, default=500)
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--reader-model", default=settings.ollama_model)
    parser.add_argument("--reader-max-tokens", type=int, default=64)
    parser.add_argument("--reader-timeout", type=int, default=300)
    parser.add_argument("--reader-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(args.max_rows_per_dataset, args.context_character_budget, args.log_every) < 1:
        parser.error("rows/context/log 값은 1 이상이어야 합니다.")
    return args


def _manifest_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    return all(saved.get(key) == value for key, value in expected.items())


def _write_manifest(path: Path, config: dict, status: str, completed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**config, "status": status, "completed_rows": completed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_dataset(args, dataset, reader, reader_config, shared_model, shared_reranker):
    joined, _ = _load_split(
        args, dataset, args.split, Path(args.trajectory_root), args.max_rows_per_dataset
    )
    full_joined, _ = _load_split(args, dataset, args.split, Path(args.trajectory_root), 2000)
    full_examples = list({example.policy_id: example for _, example in full_joined}.values())
    retriever = _load_fixed_retriever(
        Path(args.index_root) / args.split / dataset,
        _corpus_fingerprint(full_examples),
        embedding_model=shared_model,
        reranker=shared_reranker,
    )
    chunks = {chunk.chunk_id: chunk for chunk in retriever.chunks}
    policy_by_id = {
        str(row["id"]): row
        for row in _iter_jsonl(Path(args.policy_predictions))
        if str(row.get("dataset")) == dataset
    }
    direct_by_id = {
        str(row["id"]): row
        for row in _iter_jsonl(Path(args.direct_answer_root) / f"{dataset}_{args.split}.jsonl")
    }
    fixed_by_id = {
        str(row["id"]): row
        for row in _iter_jsonl(Path(args.fixed_proof_root) / f"{dataset}_{args.split}.jsonl")
    }
    output = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
    manifest = output.with_suffix(".manifest.json")
    config = {
        "format_version": FORMAT_VERSION,
        "experiment": "adaptive_proof_completion_reader",
        "dataset": dataset,
        "split": args.split,
        "requested_rows": len(joined),
        "policy_predictions": str(Path(args.policy_predictions)),
        "direct_answer_root": str(Path(args.direct_answer_root)),
        "fixed_proof_root": str(Path(args.fixed_proof_root)),
        "context_character_budget": args.context_character_budget,
        "reader": asdict(reader_config),
    }
    cached = {}
    if output.exists() or manifest.exists():
        if args.overwrite:
            cached = {}
        elif _manifest_matches(manifest, config):
            cached = {str(row["id"]): row for row in _iter_jsonl(output)} if output.exists() else {}
        else:
            raise SystemExit(f"기존 adaptive proof reader 설정과 다릅니다: {output}")
    expected_ids = {example.policy_id for _, example in joined}
    rows_by_id = {key: row for key, row in cached.items() if key in expected_ids}
    if len(rows_by_id) == len(joined):
        return [rows_by_id[example.policy_id] for _, example in joined], retriever.model, retriever.reranker
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    _write_manifest(manifest, config, "in_progress", len(rows_by_id))
    started = __import__("time").perf_counter()
    new = 0
    with output.open(mode, encoding="utf-8") as destination:
        for _, example in joined:
            if example.policy_id in rows_by_id:
                continue
            policy = policy_by_id[example.policy_id]
            sources = [
                (chunks[str(item["chunk_id"])], float(item["score"]))
                for item in policy["evidence"]
                if str(item["chunk_id"]) in chunks
            ]
            context, provenance = evidence_context(sources, args.context_character_budget)
            answer = reader.answer(example.question, context)
            score = answer_score(
                answer.prediction,
                str(example.answer),
                example.answer_aliases,
                dataset=dataset,
            )
            direct = direct_by_id[example.policy_id]["actions"]["vector"]
            fixed = fixed_by_id[example.policy_id]
            row = {
                "format_version": FORMAT_VERSION,
                "id": example.policy_id,
                "dataset": dataset,
                "question": example.question,
                "reasoning_type": example.reasoning_type,
                "prediction": answer.prediction,
                "em": score.em,
                "f1": score.f1,
                "chosen_step": int(policy["chosen_step"]),
                "complete_probability": float(policy["complete_probability"]),
                "direct_prediction": direct["prediction"],
                "direct_em": float(direct["em"]),
                "direct_f1": float(direct["f1"]),
                "fixed_prediction": fixed["prediction"],
                "fixed_em": float(fixed["em"]),
                "fixed_f1": float(fixed["f1"]),
                "evidence": list(provenance),
                "cache_hit": answer.cache_hit,
                "latency_seconds": answer.latency_seconds,
            }
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows_by_id[example.policy_id] = row
            new += 1
            completed = len(rows_by_id)
            if completed % args.log_every == 0 or completed == len(joined):
                current = list(rows_by_id.values())
                elapsed = __import__("time").perf_counter() - started
                eta = elapsed / max(1, new) * (len(joined) - completed)
                print(
                    f"{dataset} adaptive {completed}/{len(joined)} | ETA {_duration(eta)} | "
                    f"direct/fixed/adaptive F1="
                    f"{np.mean([row['direct_f1'] for row in current])*100:.2f}/"
                    f"{np.mean([row['fixed_f1'] for row in current])*100:.2f}/"
                    f"{np.mean([row['f1'] for row in current])*100:.2f} | searches="
                    f"{np.mean([row['chosen_step'] for row in current]):.2f}",
                    flush=True,
                )
                _write_manifest(manifest, config, "in_progress", completed)
    _write_manifest(manifest, config, "ready", len(rows_by_id))
    return [rows_by_id[example.policy_id] for _, example in joined], retriever.model, retriever.reranker


def main() -> None:
    args = _arguments()
    session = requests.Session()
    model, digest = ollama_model_identity(args.ollama_host, args.reader_model, session=session)
    reader_config = ReaderConfig(
        host=args.ollama_host,
        model=model,
        model_digest=digest,
        max_tokens=args.reader_max_tokens,
        timeout_seconds=args.reader_timeout,
        seed=args.reader_seed,
        temperature=0.0,
        keep_alive=args.ollama_keep_alive,
        prompt_version=PROMPT_VERSION,
        thinking=False,
    )
    shared_model = shared_reranker = None
    with AnswerCache(Path(args.cache_path)) as cache:
        reader = FrozenOllamaReader(reader_config, cache, session=session)
        for dataset in args.datasets:
            rows, shared_model, shared_reranker = _run_dataset(
                args, dataset, reader, reader_config, shared_model, shared_reranker
            )
            print(
                f"{dataset}: n={len(rows)} direct EM/F1="
                f"{np.mean([row['direct_em'] for row in rows])*100:.2f}/"
                f"{np.mean([row['direct_f1'] for row in rows])*100:.2f} fixed="
                f"{np.mean([row['fixed_em'] for row in rows])*100:.2f}/"
                f"{np.mean([row['fixed_f1'] for row in rows])*100:.2f} adaptive="
                f"{np.mean([row['em'] for row in rows])*100:.2f}/"
                f"{np.mean([row['f1'] for row in rows])*100:.2f} searches="
                f"{np.mean([row['chosen_step'] for row in rows]):.2f}",
                flush=True,
            )
    session.close()


if __name__ == "__main__":
    main()
