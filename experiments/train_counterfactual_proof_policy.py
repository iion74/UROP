"""Train and audit a counterfactual proof-completion evidence policy.

Training states are synthesized by deleting gold proof documents and adding a
hard retrieved distractor.  At every state, document actions that restore an
uncovered support group are positive; the ANSWER action is positive only for a
complete proof.  Gold annotations never enter inference features.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import chain
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments.train_multihop_policy import (
    SUPPORTED_DATASETS,
    _dataset_seed,
    _iter_jsonl,
    load_processed_examples,
)
from src.causal_evidence import POLICY_FEATURE_NAMES, proof_policy_features, run_proof_completion_policy


ACTION = "planned_continue_3"
FORMAT_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual proof-completion policy 학습/검증")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument(
        "--train-trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_train_1000",
    )
    parser.add_argument(
        "--validation-trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument("--model-out", default="data/models/counterfactual_proof_completion_policy.joblib")
    parser.add_argument("--report-out", default="data/models/counterfactual_proof_completion_policy.report.json")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-train-rows-per-dataset", type=int, default=1000)
    parser.add_argument("--max-validation-rows-per-dataset", type=int, default=2000)
    parser.add_argument("--max-selected-documents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(
        args.corpus_examples_per_dataset,
        args.max_train_rows_per_dataset,
        args.max_validation_rows_per_dataset,
        args.max_selected_documents,
    ) < 1:
        parser.error("corpus/train/validation/selected 크기는 1 이상이어야 합니다.")
    return args


def _load_split(args, dataset: str, split: str, trajectory_root: Path, maximum: int):
    examples = load_processed_examples(
        Path(args.processed_root) / dataset / f"{split}.jsonl",
        args.corpus_examples_per_dataset,
        _dataset_seed(args.corpus_seed, dataset),
    )
    examples_by_id = {example.policy_id: example for example in examples}
    documents = {
        document.content_id: document
        for example in examples
        for document in example.documents
    }
    rows = list(_iter_jsonl(trajectory_root / f"{dataset}_{split}.jsonl"))[:maximum]
    joined = [(row, examples_by_id[str(row["id"])]) for row in rows if str(row["id"]) in examples_by_id]
    if len(joined) != len(rows):
        raise SystemExit(f"{dataset}/{split}: trajectory와 processed cohort ID가 일치하지 않습니다.")
    return joined, documents


def _candidates(row: dict) -> list[dict]:
    return list(row["actions"][ACTION]["evidence"])


def _group_indices(example, candidates: list[dict]) -> list[tuple[int, ...]]:
    content_ids = [str(candidate["content_id"]) for candidate in candidates]
    return [
        tuple(index for index, content_id in enumerate(content_ids) if content_id in group)
        for group in example.supporting_groups
    ]


def _complete(selected: set[int], group_indices: list[tuple[int, ...]]) -> bool:
    return bool(group_indices) and all(selected.intersection(group) for group in group_indices)


def _training_states(candidates: list[dict], group_indices: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    representatives = [
        max(group, key=lambda index: float(candidates[index].get("score", 0.0)))
        for group in group_indices
    ]
    ordered = sorted(set(representatives), key=lambda index: -float(candidates[index].get("score", 0.0)))
    states: set[tuple[int, ...]] = {(), tuple(sorted(ordered))}
    for length in range(1, len(ordered)):
        states.add(tuple(sorted(ordered[:length])))
    for omitted in ordered:
        states.add(tuple(sorted(index for index in ordered if index != omitted)))

    support_indices = set(chain.from_iterable(group_indices))
    distractors = [index for index in range(len(candidates)) if index not in support_indices]
    if distractors:
        hard = max(distractors, key=lambda index: float(candidates[index].get("score", 0.0)))
        states.update(tuple(sorted((*state, hard))) for state in tuple(states) if hard not in state)
    return sorted(states, key=lambda state: (len(state), state))


def _build_training_arrays(args):
    feature_blocks: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    audit = {}
    for dataset in args.datasets:
        joined, documents = _load_split(
            args,
            dataset,
            "train",
            Path(args.train_trajectory_root),
            args.max_train_rows_per_dataset,
        )
        used = states = 0
        positives = Counter()
        for row, example in joined:
            candidates = _candidates(row)
            group_indices = _group_indices(example, candidates)
            if not group_indices or any(not group for group in group_indices):
                continue
            used += 1
            for selected_tuple in _training_states(candidates, group_indices):
                selected = set(selected_tuple)
                matrix = proof_policy_features(example.question, candidates, documents, selected_tuple)
                complete = _complete(selected, group_indices)
                row_labels = []
                for index in range(len(candidates)):
                    restores = index not in selected and any(
                        not selected.intersection(group) and index in group for group in group_indices
                    )
                    row_labels.append(float(restores))
                row_labels.append(float(complete))
                feature_blocks.append(matrix)
                labels.extend(row_labels)
                for index, label in enumerate(row_labels):
                    if label:
                        weights.append(1.0)
                        positives["ANSWER" if index == len(candidates) else "RETRIEVE"] += 1
                    elif index == len(candidates):
                        weights.append(1.0)
                    else:
                        weights.append(0.25 + max(0.0, float(candidates[index].get("score", 0.0))))
                states += 1
        audit[dataset] = {
            "trajectory_rows": len(joined),
            "fully_retrieved_training_rows": used,
            "counterfactual_states": states,
            "positive_actions": dict(positives),
        }
        print(f"{dataset}: usable={used:,}, counterfactual states={states:,}, positives={dict(positives)}", flush=True)
    return (
        np.vstack(feature_blocks),
        np.asarray(labels, dtype="int8"),
        np.asarray(weights, dtype="float64"),
        audit,
    )


def _support_metrics(selected: set[int], group_indices: list[tuple[int, ...]]) -> tuple[float, float, float]:
    recovered = sum(bool(selected.intersection(group)) for group in group_indices)
    recall = recovered / max(1, len(group_indices))
    support_indices = set(chain.from_iterable(group_indices))
    precision = len(selected & support_indices) / max(1, len(selected))
    return recall, float(recovered == len(group_indices)), precision


def _evaluate(args, model):
    report = {}
    predictions = []
    for dataset in args.datasets:
        joined, documents = _load_split(
            args,
            dataset,
            "validation",
            Path(args.validation_trajectory_root),
            args.max_validation_rows_per_dataset,
        )
        rows = []
        for row, example in joined:
            candidates = _candidates(row)
            groups = _group_indices(example, candidates)
            selected_tuple, trace = run_proof_completion_policy(
                model,
                example.question,
                candidates,
                documents,
                max_documents=args.max_selected_documents,
            )
            selected = set(selected_tuple)
            learned = _support_metrics(selected, groups)
            raw_top4 = _support_metrics(set(range(min(4, len(candidates)))), groups)
            all_evidence = _support_metrics(set(range(len(candidates))), groups)
            record = {
                "id": example.policy_id,
                "dataset": dataset,
                "selected_indices": list(selected_tuple),
                "selected_content_ids": [candidates[index]["content_id"] for index in selected_tuple],
                "selected_documents": len(selected_tuple),
                "answered": bool(trace and trace[-1]["action"] == "ANSWER"),
                "source_recall": learned[0],
                "all_support_recall": learned[1],
                "support_precision": learned[2],
                "top4_source_recall": raw_top4[0],
                "top4_all_support_recall": raw_top4[1],
                "all_evidence_source_recall": all_evidence[0],
                "all_evidence_all_support_recall": all_evidence[1],
                "policy_trace": list(trace),
            }
            rows.append(record)
            predictions.append(record)
        report[dataset] = {
            "n": len(rows),
            "selected_documents": float(np.mean([row["selected_documents"] for row in rows])),
            "answer_action_rate": float(np.mean([row["answered"] for row in rows])),
            "source_recall": float(np.mean([row["source_recall"] for row in rows])),
            "all_support_recall": float(np.mean([row["all_support_recall"] for row in rows])),
            "support_precision": float(np.mean([row["support_precision"] for row in rows])),
            "top4_source_recall": float(np.mean([row["top4_source_recall"] for row in rows])),
            "top4_all_support_recall": float(np.mean([row["top4_all_support_recall"] for row in rows])),
            "all_evidence_source_recall": float(np.mean([row["all_evidence_source_recall"] for row in rows])),
            "all_evidence_all_support_recall": float(np.mean([row["all_evidence_all_support_recall"] for row in rows])),
        }
        print(json.dumps({dataset: report[dataset]}, ensure_ascii=False, indent=2), flush=True)
    return report, predictions


def main() -> None:
    args = _arguments()
    features, labels, weights, training_audit = _build_training_arrays(args)
    print(f"policy actions={len(labels):,}, features={features.shape[1]}, positive={labels.mean():.2%}", flush=True)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(features, labels, sample_weight=weights)
    report, predictions = _evaluate(args, model)

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "format_version": FORMAT_VERSION,
            "method": "counterfactual_proof_completion_policy",
            "feature_names": POLICY_FEATURE_NAMES,
            "model": model,
            "max_selected_documents": args.max_selected_documents,
        },
        model_path,
    )
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = report_path.with_name(f"{report_path.stem}.predictions.jsonl")
    with prediction_path.open("w", encoding="utf-8") as destination:
        for row in predictions:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report_path.write_text(
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "method": "counterfactual_proof_completion_policy",
                "model": str(model_path),
                "features": list(POLICY_FEATURE_NAMES),
                "training": training_audit,
                "validation": report,
                "predictions": str(prediction_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {model_path}\nreport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
