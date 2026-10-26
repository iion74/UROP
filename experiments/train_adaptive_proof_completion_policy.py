"""Train an adaptive proof-completeness policy over retrieval prefixes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments.evaluate_answer_oracle import _load_fixed_retriever
from experiments.train_counterfactual_proof_policy import _load_split
from experiments.train_graph_repair_policy import _base_selection
from experiments.train_multihop_policy import (
    SUPPORTED_DATASETS,
    _corpus_fingerprint,
    _dataset_seed,
    load_processed_examples,
)
from src.causal_evidence import WIDTH_FEATURE_NAMES, evidence_width_features
from src.proof_repair import (
    build_graph_repair_proposals,
    graph_repair_features,
    repair_syndrome_features,
)


ACTIONS = ("vector", "planned_continue_1", "planned_continue_2", "planned_continue_3")
FORMAT_VERSION = 1
COMPLETENESS_FEATURE_NAMES = (
    *WIDTH_FEATURE_NAMES,
    "search_step",
    "candidate_count",
    "selected_document_count",
    "selected_support_probability_max",
    "selected_support_probability_mean",
    "selected_support_probability_min",
    "new_candidate_fraction",
    "selected_survival_fraction",
    "repair_syndrome_probability",
    "repair_rank_score_max",
    "repair_rank_score_gap",
    "repair_applied",
)


@dataclass
class PrefixState:
    dataset: str
    example_id: str
    reasoning_type: str | None
    search_step: int
    action: str
    features: np.ndarray
    complete: int
    source_recall: float
    support_precision: float
    evidence: list[dict]
    tune: bool


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive proof completeness policy")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument(
        "--train-trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_train_1000",
    )
    parser.add_argument(
        "--validation-trajectory-root",
        default="data/benchmarks/multihop/sequential_search_v6_evidence_survival_validation_2000",
    )
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--support-model", default="data/models/counterfactual_proof_completion_policy.joblib")
    parser.add_argument("--width-model", default="data/models/evidence_width_policy.joblib")
    parser.add_argument("--repair-model", default="data/models/factorized_graph_proof_repair_policy.joblib")
    parser.add_argument("--model-out", default="data/models/adaptive_proof_completion_policy.joblib")
    parser.add_argument("--report-out", default="data/models/adaptive_proof_completion_policy.report.json")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-train-rows-per-dataset", type=int, default=1000)
    parser.add_argument("--max-validation-rows-per-dataset", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(
        args.corpus_examples_per_dataset,
        args.max_train_rows_per_dataset,
        args.max_validation_rows_per_dataset,
    ) < 1:
        parser.error("corpus/train/validation 크기는 1 이상이어야 합니다.")
    return args


def _candidate_list(trajectory: dict, action: str) -> list[dict]:
    return list(trajectory["actions"][action]["evidence"])


def _repair_selection(bundle, question, candidates, selected, support_probabilities, documents, retriever):
    proposals = build_graph_repair_proposals(
        retriever,
        candidates,
        selected,
        neighbors_per_seed=int(bundle["neighbors_per_seed"]),
        maximum_proposals=int(bundle["maximum_proposals"]),
    )
    proposal_features = graph_repair_features(
        question,
        candidates,
        selected,
        support_probabilities,
        proposals,
        documents,
    )
    rank_scores = (
        bundle["ranker"].predict_proba(proposal_features)[:, 1]
        if len(proposal_features)
        else np.empty(0, dtype="float64")
    )
    syndrome_features = repair_syndrome_features(
        proposal_features, rank_scores, len(selected)
    )
    syndrome_probability = float(
        bundle["syndrome"].predict_proba(syndrome_features.reshape(1, -1))[0, 1]
    )
    repairs = (
        (int(np.argmax(rank_scores)),)
        if len(rank_scores) and syndrome_probability >= float(bundle["syndrome_threshold"])
        else ()
    )
    return proposals, rank_scores, syndrome_probability, repairs


def _support_metrics(ids: set[str], groups) -> tuple[float, int, float]:
    recovered = sum(bool(ids.intersection(group)) for group in groups)
    recall = recovered / max(1, len(groups))
    gold = set(chain.from_iterable(groups))
    precision = len(ids.intersection(gold)) / max(1, len(ids))
    return recall, int(recovered == len(groups)), precision


def _build_states(args, split, trajectory_root, maximum, support_model, width_bundle, repair_bundle):
    all_states: list[PrefixState] = []
    audit = {}
    shared_model = shared_reranker = None
    for dataset in args.datasets:
        joined, documents = _load_split(args, dataset, split, trajectory_root, maximum)
        corpus_examples = load_processed_examples(
            Path(args.processed_root) / dataset / f"{split}.jsonl",
            args.corpus_examples_per_dataset,
            _dataset_seed(args.corpus_seed, dataset),
        )
        retriever = _load_fixed_retriever(
            Path(args.index_root) / split / dataset,
            _corpus_fingerprint(corpus_examples),
            embedding_model=shared_model,
            reranker=shared_reranker,
        )
        shared_model = retriever.model or shared_model
        shared_reranker = retriever.reranker or shared_reranker
        complete_by_step = Counter()
        repair_by_step = Counter()
        for position, (trajectory, example) in enumerate(joined):
            previous_candidates: set[str] = set()
            previous_selected: set[str] = set()
            for search_step, action in enumerate(ACTIONS):
                candidates = _candidate_list(trajectory, action)
                selected, support_probabilities = _base_selection(
                    support_model, width_bundle, example.question, candidates, documents
                )
                proposals, rank_scores, syndrome_probability, repairs = _repair_selection(
                    repair_bundle,
                    example.question,
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
                width_features = evidence_width_features(
                    support_model, example.question, candidates, documents
                )
                feature_values = np.asarray(
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
                        (
                            float(np.partition(rank_scores, -2)[-1] - np.partition(rank_scores, -2)[-2])
                            if len(rank_scores) > 1
                            else max(rank_scores, default=0.0)
                        ),
                        float(bool(repairs)),
                    ],
                    dtype="float64",
                )
                if len(feature_values) != len(COMPLETENESS_FEATURE_NAMES):
                    raise RuntimeError("proof completeness feature schema mismatch")
                recall, complete, precision = _support_metrics(
                    evidence_ids, example.supporting_groups
                )
                complete_by_step[search_step] += complete
                repair_by_step[search_step] += bool(repairs)
                all_states.append(
                    PrefixState(
                        dataset=dataset,
                        example_id=example.policy_id,
                        reasoning_type=example.reasoning_type,
                        search_step=search_step,
                        action=action,
                        features=feature_values,
                        complete=complete,
                        source_recall=recall,
                        support_precision=precision,
                        evidence=evidence,
                        tune=position % 5 == 0,
                    )
                )
                previous_candidates = current_candidates
                previous_selected = evidence_ids
        audit[dataset] = {
            "trajectories": len(joined),
            "states": len(joined) * len(ACTIONS),
            "complete_by_step": {str(step): complete_by_step[step] for step in range(len(ACTIONS))},
            "repair_by_step": {str(step): repair_by_step[step] for step in range(len(ACTIONS))},
        }
        print(json.dumps({dataset: audit[dataset]}, ensure_ascii=False), flush=True)
    return all_states, audit


def _fit(states, seed):
    fit_states = [state for state in states if not state.tune]
    features = np.vstack([state.features for state in fit_states])
    labels = np.asarray([state.complete for state in fit_states], dtype="int8")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=2.0,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(features, labels)
    return model, {
        "states": len(labels),
        "complete_states": int(labels.sum()),
        "complete_rate": float(labels.mean()),
    }


def _group_trajectories(states):
    grouped = defaultdict(list)
    for state in states:
        grouped[(state.dataset, state.example_id)].append(state)
    return [sorted(group, key=lambda state: state.search_step) for group in grouped.values()]


def _evaluate_threshold(trajectories, model, threshold):
    rows = []
    for trajectory in trajectories:
        probabilities = model.predict_proba(np.vstack([state.features for state in trajectory]))[:, 1]
        chosen_index = next(
            (index for index, probability in enumerate(probabilities) if probability >= threshold),
            len(trajectory) - 1,
        )
        chosen = trajectory[chosen_index]
        final = trajectory[-1]
        rows.append(
            {
                "dataset": chosen.dataset,
                "chosen_step": chosen.search_step,
                "source_recall": chosen.source_recall,
                "all_support_recall": chosen.complete,
                "support_precision": chosen.support_precision,
                "fixed_source_recall": final.source_recall,
                "fixed_all_support_recall": final.complete,
                "complete_probability": float(probabilities[chosen_index]),
            }
        )
    report = {}
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        subset = [row for row in rows if row["dataset"] == dataset]
        report[dataset] = {
            "count": len(subset),
            "average_additional_searches": float(np.mean([row["chosen_step"] for row in subset])),
            "source_recall": float(np.mean([row["source_recall"] for row in subset])),
            "all_support_recall": float(np.mean([row["all_support_recall"] for row in subset])),
            "support_precision": float(np.mean([row["support_precision"] for row in subset])),
            "fixed_source_recall": float(np.mean([row["fixed_source_recall"] for row in subset])),
            "fixed_all_support_recall": float(np.mean([row["fixed_all_support_recall"] for row in subset])),
            "step_distribution": dict(sorted(Counter(row["chosen_step"] for row in subset).items())),
        }
    return report, rows


def _choose_threshold(states, model):
    tuning = _group_trajectories([state for state in states if state.tune])
    sweep = []
    thresholds = np.unique(
        np.concatenate(
            (
                np.linspace(0.10, 0.95, 35),
                np.asarray([0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999]),
            )
        )
    )
    for threshold in thresholds:
        report, _ = _evaluate_threshold(tuning, model, float(threshold))
        losses = {
            dataset: (result["all_support_recall"] - result["fixed_all_support_recall"]) * 100.0
            for dataset, result in report.items()
        }
        average_steps = {dataset: result["average_additional_searches"] for dataset, result in report.items()}
        sweep.append(
            {
                "threshold": float(threshold),
                "datasets": report,
                "minimum_all_support_change_points": min(losses.values()),
                "maximum_average_searches": max(average_steps.values()),
                "sum_average_searches": sum(average_steps.values()),
            }
        )
    acceptable = [row for row in sweep if row["minimum_all_support_change_points"] >= -1.0]
    if acceptable:
        chosen = min(
            acceptable,
            key=lambda row: (
                row["sum_average_searches"],
                -row["minimum_all_support_change_points"],
                row["threshold"],
            ),
        )
    else:
        chosen = max(
            sweep,
            key=lambda row: (
                row["minimum_all_support_change_points"],
                -row["sum_average_searches"],
                row["threshold"],
            ),
        )
    return chosen, sweep


def _validation_predictions(states, model, threshold):
    trajectories = _group_trajectories(states)
    report, summary_rows = _evaluate_threshold(trajectories, model, threshold)
    output_rows = []
    for trajectory, summary in zip(trajectories, summary_rows, strict=True):
        probabilities = model.predict_proba(np.vstack([state.features for state in trajectory]))[:, 1]
        chosen_index = next(
            (index for index, probability in enumerate(probabilities) if probability >= threshold),
            len(trajectory) - 1,
        )
        chosen = trajectory[chosen_index]
        output_rows.append(
            {
                "format_version": FORMAT_VERSION,
                "id": chosen.example_id,
                "dataset": chosen.dataset,
                "reasoning_type": chosen.reasoning_type,
                "chosen_step": chosen.search_step,
                "chosen_action": chosen.action,
                "complete_probability": float(probabilities[chosen_index]),
                "probabilities_by_step": [float(value) for value in probabilities],
                "source_recall": chosen.source_recall,
                "all_support_recall": chosen.complete,
                "support_precision": chosen.support_precision,
                "evidence": chosen.evidence,
            }
        )
    return report, output_rows


def main() -> None:
    args = _arguments()
    support_model = joblib.load(args.support_model)["model"]
    width_bundle = joblib.load(args.width_model)
    repair_bundle = joblib.load(args.repair_model)
    train_states, training_audit = _build_states(
        args,
        "train",
        Path(args.train_trajectory_root),
        args.max_train_rows_per_dataset,
        support_model,
        width_bundle,
        repair_bundle,
    )
    model, fit_report = _fit(train_states, args.seed)
    chosen, threshold_sweep = _choose_threshold(train_states, model)
    print(
        f"chosen threshold={chosen['threshold']:.3f}, minimum all-support change="
        f"{chosen['minimum_all_support_change_points']:+.2f}, searches={chosen['sum_average_searches']:.2f}",
        flush=True,
    )
    validation_states, validation_audit = _build_states(
        args,
        "validation",
        Path(args.validation_trajectory_root),
        args.max_validation_rows_per_dataset,
        support_model,
        width_bundle,
        repair_bundle,
    )
    validation, predictions = _validation_predictions(
        validation_states, model, chosen["threshold"]
    )
    for dataset, result in validation.items():
        print(
            f"{dataset}: searches={result['average_additional_searches']:.2f}, all-support="
            f"{result['fixed_all_support_recall']*100:.2f}->{result['all_support_recall']*100:.2f}, "
            f"source={result['fixed_source_recall']*100:.2f}->{result['source_recall']*100:.2f}, "
            f"steps={result['step_distribution']}",
            flush=True,
        )
    bundle = {
        "format_version": FORMAT_VERSION,
        "method": "adaptive_proof_completeness_policy",
        "feature_names": COMPLETENESS_FEATURE_NAMES,
        "model": model,
        "threshold": chosen["threshold"],
        "maximum_additional_searches": len(ACTIONS) - 1,
        "support_model": str(Path(args.support_model)),
        "width_model": str(Path(args.width_model)),
        "repair_model": str(Path(args.repair_model)),
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = report_path.with_name(f"{report_path.stem}.predictions.jsonl")
    with predictions_path.open("w", encoding="utf-8") as destination:
        for row in predictions:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "format_version": FORMAT_VERSION,
        "method": bundle["method"],
        "model": str(model_path),
        "predictions": str(predictions_path),
        "training_audit": training_audit,
        "fit": fit_report,
        "chosen_threshold": chosen,
        "threshold_sweep": threshold_sweep,
        "validation_audit": validation_audit,
        "validation": validation,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {model_path}\nreport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
