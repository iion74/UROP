"""Evaluate every compact-proof prefix with an explicit evidence-chain reader.

The resulting rows are used twice: train rows provide offline STOP/CONTINUE
targets, while validation rows are consumed only after the policy is frozen.
All policy-facing features are observable at inference time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import requests

from experiments.evaluate_answer_oracle import _load_fixed_retriever
from experiments.train_adaptive_proof_completion_policy import (
    ACTIONS,
    COMPLETENESS_FEATURE_NAMES,
    _candidate_list,
    _repair_selection,
    _support_metrics,
)
from experiments.train_counterfactual_proof_policy import _load_split
from experiments.train_graph_repair_policy import _base_selection
from experiments.train_multihop_policy import (
    SUPPORTED_DATASETS,
    _corpus_fingerprint,
    _dataset_seed,
    _duration,
    _iter_jsonl,
    load_processed_examples,
)
from src.causal_evidence import evidence_width_features
from src.qa_evaluation import (
    EVIDENCE_CHAIN_PROMPT_VERSION,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    answer_score,
    evidence_context,
    ollama_model_identity,
)
from src.reader_sufficiency import (
    READER_SUFFICIENCY_FEATURE_NAMES,
    parse_evidence_chain,
    reader_sufficiency_features,
)
from src.settings import settings


FORMAT_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reader-aware compact proof prefix evaluation")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--support-model", default="data/models/counterfactual_proof_completion_policy.joblib")
    parser.add_argument("--width-model", default="data/models/evidence_width_policy.joblib")
    parser.add_argument("--repair-model", default="data/models/factorized_graph_proof_repair_policy.joblib")
    parser.add_argument(
        "--completeness-model",
        default="data/models/adaptive_proof_completion_policy_high_threshold.joblib",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-path", default="data/benchmarks/multihop/answer_cache.sqlite3")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-rows-per-dataset", type=int, required=True)
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--reader-model", default=settings.ollama_model)
    parser.add_argument("--reader-max-tokens", type=int, default=256)
    parser.add_argument("--reader-timeout", type=int, default=300)
    parser.add_argument("--reader-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default=settings.ollama_keep_alive)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(
        args.corpus_examples_per_dataset,
        args.max_rows_per_dataset,
        args.context_character_budget,
        args.reader_max_tokens,
        args.log_every,
    ) < 1:
        parser.error("corpus/rows/context/tokens/log 값은 1 이상이어야 합니다.")
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


def _proof_state(
    question,
    search_step,
    candidates,
    previous_candidates,
    previous_selected,
    documents,
    retriever,
    chunks,
    support_model,
    width_bundle,
    repair_bundle,
):
    selected, support_probabilities = _base_selection(
        support_model, width_bundle, question, candidates, documents
    )
    proposals, rank_scores, syndrome_probability, repairs = _repair_selection(
        repair_bundle,
        question,
        candidates,
        selected,
        support_probabilities,
        documents,
        retriever,
    )
    evidence = [
        {
            "chunk_id": str(candidates[index]["chunk_id"]),
            "content_id": str(candidates[index]["content_id"]),
            "title": str(candidates[index].get("title", "")),
            "score": float(support_probabilities[index]),
            "kind": "selected",
        }
        for index in selected
    ]
    evidence.extend(
        {
            "chunk_id": str(proposals[index]["chunk_id"]),
            "content_id": str(proposals[index]["content_id"]),
            "title": str(proposals[index].get("title", "")),
            "score": float(rank_scores[index]),
            "kind": "repair",
        }
        for index in repairs
    )
    evidence_ids = {str(item["content_id"]) for item in evidence}
    current_candidates = {str(candidate["content_id"]) for candidate in candidates}
    selected_probabilities = [float(support_probabilities[index]) for index in selected]
    width_features = evidence_width_features(support_model, question, candidates, documents)
    rank_gap = (
        float(np.partition(rank_scores, -2)[-1] - np.partition(rank_scores, -2)[-2])
        if len(rank_scores) > 1
        else max(rank_scores, default=0.0)
    )
    features = np.asarray(
        [
            *width_features.tolist(),
            float(search_step),
            float(len(candidates)),
            float(len(evidence)),
            max(selected_probabilities, default=0.0),
            float(np.mean(selected_probabilities)) if selected_probabilities else 0.0,
            min(selected_probabilities, default=0.0),
            len(current_candidates - previous_candidates) / max(1, len(current_candidates)),
            len(evidence_ids.intersection(previous_selected)) / max(1, len(previous_selected)),
            syndrome_probability,
            max(rank_scores, default=0.0),
            rank_gap,
            float(bool(repairs)),
        ],
        dtype="float64",
    )
    if len(features) != len(COMPLETENESS_FEATURE_NAMES):
        raise RuntimeError("proof completeness feature schema mismatch")
    sources = [
        (chunks[str(item["chunk_id"])], float(item["score"]))
        for item in evidence
        if str(item["chunk_id"]) in chunks
    ]
    return evidence, sources, features, current_candidates, evidence_ids


def _run_dataset(
    args,
    dataset,
    reader,
    reader_config,
    support_model,
    width_bundle,
    repair_bundle,
    completeness_bundle,
    shared_model,
    shared_reranker,
):
    joined, documents = _load_split(
        args, dataset, args.split, Path(args.trajectory_root), args.max_rows_per_dataset
    )
    corpus_examples = load_processed_examples(
        Path(args.processed_root) / dataset / f"{args.split}.jsonl",
        args.corpus_examples_per_dataset,
        _dataset_seed(args.corpus_seed, dataset),
    )
    retriever = _load_fixed_retriever(
        Path(args.index_root) / args.split / dataset,
        _corpus_fingerprint(corpus_examples),
        embedding_model=shared_model,
        reranker=shared_reranker,
    )
    chunks = {chunk.chunk_id: chunk for chunk in retriever.chunks}
    output = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
    manifest = output.with_suffix(".manifest.json")
    config = {
        "format_version": FORMAT_VERSION,
        "experiment": "reader_aware_proof_prefixes",
        "dataset": dataset,
        "split": args.split,
        "requested_rows": len(joined),
        "actions": list(ACTIONS),
        "support_model": str(Path(args.support_model)),
        "width_model": str(Path(args.width_model)),
        "repair_model": str(Path(args.repair_model)),
        "completeness_model": str(Path(args.completeness_model)),
        "context_character_budget": args.context_character_budget,
        "feature_names": [
            *COMPLETENESS_FEATURE_NAMES,
            "proof_complete_probability",
            *READER_SUFFICIENCY_FEATURE_NAMES,
        ],
        "reader": asdict(reader_config),
    }
    cached = {}
    if output.exists() or manifest.exists():
        if args.overwrite:
            cached = {}
        elif _manifest_matches(manifest, config):
            cached = {str(row["id"]): row for row in _iter_jsonl(output)} if output.exists() else {}
        else:
            raise SystemExit(f"기존 reader-aware prefix 설정과 다릅니다: {output}")
    expected_ids = {example.policy_id for _, example in joined}
    rows_by_id = {key: row for key, row in cached.items() if key in expected_ids}
    if len(rows_by_id) == len(joined):
        return [rows_by_id[example.policy_id] for _, example in joined], retriever.model, retriever.reranker

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    _write_manifest(manifest, config, "in_progress", len(rows_by_id))
    started = perf_counter()
    new = 0
    with output.open(mode, encoding="utf-8") as destination:
        for trajectory, example in joined:
            if example.policy_id in rows_by_id:
                continue
            previous_candidates: set[str] = set()
            previous_selected: set[str] = set()
            prior_predictions: list[str] = []
            prior_chains: list[tuple[str, ...]] = []
            action_rows = {}
            for search_step, action in enumerate(ACTIONS):
                candidates = _candidate_list(trajectory, action)
                evidence, sources, proof_features, current_candidates, evidence_ids = _proof_state(
                    example.question,
                    search_step,
                    candidates,
                    previous_candidates,
                    previous_selected,
                    documents,
                    retriever,
                    chunks,
                    support_model,
                    width_bundle,
                    repair_bundle,
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
                complete_probability = float(
                    completeness_bundle["model"].predict_proba(proof_features.reshape(1, -1))[0, 1]
                )
                policy_features = np.concatenate(
                    (proof_features, np.asarray([complete_probability]), reader_features)
                )
                score = answer_score(
                    result.prediction,
                    str(example.answer),
                    example.answer_aliases,
                    dataset=dataset,
                )
                recall, complete, precision = _support_metrics(evidence_ids, example.supporting_groups)
                action_rows[action] = {
                    "search_step": search_step,
                    "prediction": result.prediction,
                    "evidence_chain": list(chain),
                    "em": score.em,
                    "f1": score.f1,
                    "source_recall": recall,
                    "all_support_recall": complete,
                    "support_precision": precision,
                    "complete_probability": complete_probability,
                    "policy_features": policy_features.tolist(),
                    "evidence": list(provenance),
                    "context_characters": len(context),
                    "cache_hit": result.cache_hit,
                    "latency_seconds": result.latency_seconds,
                }
                previous_candidates = current_candidates
                previous_selected = evidence_ids
                prior_predictions.append(result.prediction)
                prior_chains.append(chain)
            row = {
                "format_version": FORMAT_VERSION,
                "id": example.policy_id,
                "dataset": dataset,
                "question": example.question,
                "reasoning_type": example.reasoning_type,
                "actions": action_rows,
            }
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows_by_id[example.policy_id] = row
            new += 1
            completed = len(rows_by_id)
            if completed % args.log_every == 0 or completed == len(joined):
                current = list(rows_by_id.values())
                terminal = [row["actions"][ACTIONS[-1]] for row in current]
                elapsed = perf_counter() - started
                eta = elapsed / max(1, new) * (len(joined) - completed)
                cache_hits = sum(
                    action["cache_hit"] for row in current for action in row["actions"].values()
                )
                print(
                    f"{dataset} reader-aware {completed}/{len(joined)} | ETA {_duration(eta)} | "
                    f"fixed3 EM/F1={np.mean([r['em'] for r in terminal])*100:.2f}/"
                    f"{np.mean([r['f1'] for r in terminal])*100:.2f} | cache={cache_hits}/{len(current)*4}",
                    flush=True,
                )
                _write_manifest(manifest, config, "in_progress", completed)
    _write_manifest(manifest, config, "ready", len(rows_by_id))
    return [rows_by_id[example.policy_id] for _, example in joined], retriever.model, retriever.reranker


def main() -> None:
    args = _arguments()
    support_model = joblib.load(args.support_model)["model"]
    width_bundle = joblib.load(args.width_model)
    repair_bundle = joblib.load(args.repair_model)
    completeness_bundle = joblib.load(args.completeness_model)
    expected_names = tuple(completeness_bundle.get("feature_names", ()))
    if expected_names and expected_names != COMPLETENESS_FEATURE_NAMES:
        raise SystemExit("completeness model feature schema가 현재 코드와 다릅니다.")
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
        prompt_version=EVIDENCE_CHAIN_PROMPT_VERSION,
        thinking=False,
    )
    shared_model = shared_reranker = None
    with AnswerCache(Path(args.cache_path)) as cache:
        reader = FrozenOllamaReader(reader_config, cache, session=session)
        for dataset in args.datasets:
            rows, shared_model, shared_reranker = _run_dataset(
                args,
                dataset,
                reader,
                reader_config,
                support_model,
                width_bundle,
                repair_bundle,
                completeness_bundle,
                shared_model,
                shared_reranker,
            )
            terminal = [row["actions"][ACTIONS[-1]] for row in rows]
            print(
                f"{dataset}: n={len(rows)} chain fixed3 EM/F1="
                f"{np.mean([row['em'] for row in terminal])*100:.2f}/"
                f"{np.mean([row['f1'] for row in terminal])*100:.2f}",
                flush=True,
            )
    session.close()


if __name__ == "__main__":
    main()
