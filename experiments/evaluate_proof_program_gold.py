"""Pilot executable proof programs on exactly the gold-support reader context."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from src.proof_program import (
    BINDING_PROMPT_VERSION,
    CONTRACT_PROMPT_VERSION,
    PROMPT_VERSION,
    FrozenProofContractReader,
    FrozenProofProgramReader,
)
from src.qa_evaluation import AnswerCache, ReaderConfig, answer_score, evidence_context, ollama_model_identity
from src.retriever import SemanticRetriever
from src.settings import settings


FORMAT_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold evidence executable proof-program pilot")
    parser.add_argument("--baseline-root", default="data/benchmarks/multihop/answer_oracle")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-path", default="data/benchmarks/multihop/proof_program_cache.sqlite3")
    parser.add_argument(
        "--reader-mode",
        choices=("direct_program", "immutable_contract"),
        default="direct_program",
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--max-rows-per-dataset", type=int, default=30)
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--reader-model", default=settings.ollama_model)
    parser.add_argument("--reader-max-tokens", type=int, default=512)
    parser.add_argument("--reader-timeout", type=int, default=300)
    parser.add_argument("--reader-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(args.max_rows_per_dataset, args.context_character_budget, args.reader_max_tokens, args.log_every) < 1:
        parser.error("rows/context/tokens/log는 1 이상이어야 합니다.")
    return args


def _manifest_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    return all(saved.get(key) == value for key, value in expected.items())


def _manifest(path: Path, config: dict, status: str, completed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**config, "status": status, "completed_rows": completed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_dataset(args, dataset: str, reader, config: ReaderConfig):
    baseline_path = Path(args.baseline_root) / f"{dataset}_{args.split}.jsonl"
    baseline_rows = list(_iter_jsonl(baseline_path))[: args.max_rows_per_dataset]
    if not baseline_rows or any("gold_support" not in row for row in baseline_rows):
        raise SystemExit(f"gold-support baseline이 없습니다: {baseline_path}")
    retriever = SemanticRetriever(Path(args.index_root) / args.split / dataset)
    if not retriever.load():
        raise SystemExit(f"index를 읽지 못했습니다: {retriever.index_dir}")
    chunks = {chunk.chunk_id: chunk for chunk in retriever.chunks}

    output = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
    manifest_path = output.with_suffix(".manifest.json")
    manifest_config = {
        "format_version": FORMAT_VERSION,
        "experiment": "executable_proof_program_gold_support",
        "reader_mode": args.reader_mode,
        "dataset": dataset,
        "split": args.split,
        "requested_rows": len(baseline_rows),
        "context_character_budget": args.context_character_budget,
        "reader": {
            **asdict(config),
            "prompt_version": (
                [CONTRACT_PROMPT_VERSION, BINDING_PROMPT_VERSION]
                if args.reader_mode == "immutable_contract"
                else PROMPT_VERSION
            ),
        },
    }
    cached = {}
    if output.exists() or manifest_path.exists():
        if args.overwrite:
            cached = {}
        elif _manifest_matches(manifest_path, manifest_config):
            cached = {str(row["id"]): row for row in _iter_jsonl(output)} if output.exists() else {}
        else:
            raise SystemExit(f"기존 proof-program 설정과 다릅니다: {output}")
    expected_ids = {str(row["id"]) for row in baseline_rows}
    rows_by_id = {row_id: row for row_id, row in cached.items() if row_id in expected_ids}
    if len(rows_by_id) == len(baseline_rows):
        return [rows_by_id[str(row["id"])] for row in baseline_rows]

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    _manifest(manifest_path, manifest_config, "in_progress", len(rows_by_id))
    started = perf_counter()
    new = 0
    with output.open(mode, encoding="utf-8") as destination:
        for baseline in baseline_rows:
            row_id = str(baseline["id"])
            if row_id in rows_by_id:
                continue
            evidence = baseline["gold_support"]["evidence"]
            sources = [
                (chunks[str(item["chunk_id"])], float(item["score"]))
                for item in evidence
                if str(item["chunk_id"]) in chunks
            ]
            context, provenance = evidence_context(sources, args.context_character_budget)
            result = reader.answer(str(baseline["question"]), context)
            score = answer_score(
                result.prediction,
                str(baseline["answer"]),
                baseline.get("answer_aliases", ()),
                dataset=dataset,
            )
            row = {
                "format_version": FORMAT_VERSION,
                "id": row_id,
                "dataset": dataset,
                "question": baseline["question"],
                "reasoning_type": baseline.get("reasoning_type"),
                "prediction": result.prediction,
                "em": score.em,
                "f1": score.f1,
                "baseline_prediction": baseline["gold_support"]["prediction"],
                "baseline_em": float(baseline["gold_support"]["em"]),
                "baseline_f1": float(baseline["gold_support"]["f1"]),
                "operation": result.operation,
                "valid_program": result.valid,
                "required_step_ids": list(result.required_step_ids),
                "counterfactual_necessity": result.counterfactual_necessity,
                "distractor_invariant": result.distractor_invariant,
                "raw_response": result.raw_response,
                "evidence": list(provenance),
                "cache_hit": result.cache_hit,
                "latency_seconds": result.latency_seconds,
            }
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows_by_id[row_id] = row
            new += 1
            completed = len(rows_by_id)
            if completed % args.log_every == 0 or completed == len(baseline_rows):
                current = list(rows_by_id.values())
                elapsed = perf_counter() - started
                eta = elapsed / max(1, new) * (len(baseline_rows) - completed)
                print(
                    f"{dataset} proof {completed}/{len(baseline_rows)} | ETA {_duration(eta)} | "
                    f"baseline/proof F1={np.mean([r['baseline_f1'] for r in current])*100:.2f}/"
                    f"{np.mean([r['f1'] for r in current])*100:.2f} | "
                    f"valid={np.mean([r['valid_program'] for r in current])*100:.1f}%",
                    flush=True,
                )
                _manifest(manifest_path, manifest_config, "in_progress", completed)
    _manifest(manifest_path, manifest_config, "ready", len(rows_by_id))
    return [rows_by_id[str(row["id"])] for row in baseline_rows]


def main() -> None:
    args = _arguments()
    session = requests.Session()
    model, digest = ollama_model_identity(args.ollama_host, args.reader_model, session=session)
    config = ReaderConfig(
        host=args.ollama_host,
        model=model,
        model_digest=digest,
        max_tokens=args.reader_max_tokens,
        timeout_seconds=args.reader_timeout,
        seed=args.reader_seed,
        temperature=0.0,
        keep_alive=args.ollama_keep_alive,
        prompt_version=(CONTRACT_PROMPT_VERSION if args.reader_mode == "immutable_contract" else PROMPT_VERSION),
        thinking=False,
    )
    with AnswerCache(Path(args.cache_path)) as cache:
        reader = (
            FrozenProofContractReader(config, cache, session=session)
            if args.reader_mode == "immutable_contract"
            else FrozenProofProgramReader(config, cache, session=session)
        )
        for dataset in args.datasets:
            rows = _run_dataset(args, dataset, reader, config)
            print(
                f"{dataset}: n={len(rows)} baseline EM/F1="
                f"{np.mean([r['baseline_em'] for r in rows])*100:.2f}/"
                f"{np.mean([r['baseline_f1'] for r in rows])*100:.2f} proof EM/F1="
                f"{np.mean([r['em'] for r in rows])*100:.2f}/"
                f"{np.mean([r['f1'] for r in rows])*100:.2f} operations="
                f"{dict(Counter(r['operation'] for r in rows))}",
                flush=True,
            )
    session.close()


if __name__ == "__main__":
    main()
