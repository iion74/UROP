"""Train a local graph-edge policy to repair missing multi-hop evidence."""

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
from experiments.train_counterfactual_proof_policy import _candidates, _load_split
from experiments.train_multihop_policy import (
    SUPPORTED_DATASETS,
    _corpus_fingerprint,
    _dataset_seed,
    load_processed_examples,
)
from src.causal_evidence import (
    evidence_width_features,
    proof_policy_features,
    select_topology_aware_indices,
)
from src.proof_repair import (
    REPAIR_FEATURE_NAMES,
    build_graph_repair_proposals,
    graph_repair_features,
    select_graph_repairs,
)


FORMAT_VERSION = 1


@dataclass
class RepairState:
    dataset: str
    example_id: str
    reasoning_type: str | None
    features: np.ndarray
    labels: np.ndarray
    proposals: list[dict]
    selected_ids: set[str]
    supporting_groups: tuple[frozenset[str], ...]
    tune: bool


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual local graph repair policy")
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
    parser.add_argument("--model-out", default="data/models/local_graph_proof_repair_policy.joblib")
    parser.add_argument("--report-out", default="data/models/local_graph_proof_repair_policy.report.json")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--corpus-examples-per-dataset", type=int, default=2000)
    parser.add_argument("--corpus-seed", type=int, default=42)
    parser.add_argument("--max-train-rows-per-dataset", type=int, default=1000)
    parser.add_argument("--max-validation-rows-per-dataset", type=int, default=2000)
    parser.add_argument("--neighbors-per-seed", type=int, default=6)
    parser.add_argument("--maximum-proposals", type=int, default=48)
    parser.add_argument("--maximum-additions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(
        args.corpus_examples_per_dataset,
        args.max_train_rows_per_dataset,
        args.max_validation_rows_per_dataset,
        args.neighbors_per_seed,
        args.maximum_proposals,
        args.maximum_additions,
    ) < 1:
        parser.error("corpus/rows/proposals/additions 값은 1 이상이어야 합니다.")
    return args


def _base_selection(support_model, width_bundle, question, candidates, documents):
    matrix = proof_policy_features(question, candidates, documents, ())
    probabilities = support_model.predict_proba(matrix)[:, 1][:-1]
    width_features = evidence_width_features(support_model, question, candidates, documents)
    raw_width = float(width_bundle["model"].predict(width_features.reshape(1, -1))[0])
    width = int(
        np.clip(
            np.ceil(raw_width),
            int(width_bundle["minimum_width"]),
            min(int(width_bundle["maximum_width"]), len(candidates)),
        )
    )
    selected = select_topology_aware_indices(question, candidates, documents, probabilities, width)
    return selected, probabilities


def _build_states(args, split, trajectory_root, maximum, support_model, width_bundle):
    states: list[RepairState] = []
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
        examples_with_positive = proposals = positives = 0
        for position, (row, example) in enumerate(joined):
            candidates = _candidates(row)
            selected, support_probabilities = _base_selection(
                support_model, width_bundle, example.question, candidates, documents
            )
            repair_proposals = build_graph_repair_proposals(
                retriever,
                candidates,
                selected,
                neighbors_per_seed=args.neighbors_per_seed,
                maximum_proposals=args.maximum_proposals,
            )
            features = graph_repair_features(
                example.question,
                candidates,
                selected,
                support_probabilities,
                repair_proposals,
                documents,
            )
            selected_ids = {str(candidates[index]["content_id"]) for index in selected}
            missing_groups = [
                group for group in example.supporting_groups if not selected_ids.intersection(group)
            ]
            labels = np.asarray(
                [
                    float(any(str(proposal["content_id"]) in group for group in missing_groups))
                    for proposal in repair_proposals
                ],
                dtype="int8",
            )
            examples_with_positive += int(bool(labels.any()))
            proposals += len(labels)
            positives += int(labels.sum())
            states.append(
                RepairState(
                    dataset=dataset,
                    example_id=example.policy_id,
                    reasoning_type=example.reasoning_type,
                    features=features,
                    labels=labels,
                    proposals=repair_proposals,
                    selected_ids=selected_ids,
                    supporting_groups=example.supporting_groups,
                    tune=position % 5 == 0,
                )
            )
        audit[dataset] = {
            "states": len(joined),
            "states_with_reachable_missing_support": examples_with_positive,
            "proposals": proposals,
            "positive_proposals": positives,
        }
        print(json.dumps({dataset: audit[dataset]}, ensure_ascii=False), flush=True)
    return states, audit


def _fit(states, *, only_fit: bool, seed: int):
    eligible = [state for state in states if len(state.labels) and (not only_fit or not state.tune)]
    features = np.vstack([state.features for state in eligible])
    labels = np.concatenate([state.labels for state in eligible])
    weights = np.where(labels > 0, 1.0, 0.25)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(features, labels, sample_weight=weights)
    return model, {"rows": len(labels), "positive_rate": float(labels.mean())}


def _state_metrics(state: RepairState, chosen: Sequence[int]) -> tuple[float, float, float, float]:
    ids = set(state.selected_ids)
    ids.update(str(state.proposals[index]["content_id"]) for index in chosen)
    recovered = sum(bool(ids.intersection(group)) for group in state.supporting_groups)
    recall = recovered / max(1, len(state.supporting_groups))
    gold_ids = set(chain.from_iterable(state.supporting_groups))
    precision = len(ids.intersection(gold_ids)) / max(1, len(ids))
    all_support = float(recovered == len(state.supporting_groups))
    harmonic = 2.0 * recall * precision / max(1e-8, recall + precision)
    return recall, all_support, precision, harmonic


def _evaluate_threshold(states, model, threshold, maximum_additions):
    metrics = []
    additions = []
    repair_labels = []
    for state in states:
        if not len(state.labels):
            continue
        probabilities = model.predict_proba(state.features)[:, 1]
        chosen = (
            ()
            if threshold >= 1.0
            else select_graph_repairs(
                probabilities,
                state.proposals,
                threshold=threshold,
                maximum_additions=maximum_additions,
            )
        )
        metrics.append(_state_metrics(state, chosen))
        additions.append(len(chosen))
        repair_labels.extend(int(state.labels[index]) for index in chosen)
    array = np.asarray(metrics, dtype="float64")
    return {
        "threshold": float(threshold),
        "source_recall": float(array[:, 0].mean()),
        "all_support_recall": float(array[:, 1].mean()),
        "support_precision": float(array[:, 2].mean()),
        "support_f1": float(array[:, 3].mean()),
        "average_additions": float(np.mean(additions)),
        "repair_precision": float(np.mean(repair_labels)) if repair_labels else 1.0,
    }


def _choose_threshold(states, model, maximum_additions):
    tuning = [state for state in states if state.tune]
    baseline = _evaluate_threshold(tuning, model, 1.0, maximum_additions)
    candidates = np.unique(
        np.concatenate(
            (
                np.linspace(0.30, 0.95, 27),
                np.asarray([0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.9925, 0.995, 0.9975, 0.999]),
            )
        )
    )
    sweep = [_evaluate_threshold(tuning, model, value, maximum_additions) for value in candidates]
    for result in sweep:
        result["utility"] = (
            result["all_support_recall"]
            + 0.50 * result["source_recall"]
            + 0.25 * result["support_precision"]
            - 0.02 * result["average_additions"]
        )
    chosen = max(
        sweep,
        key=lambda result: (
            result["utility"],
            result["repair_precision"],
            result["threshold"],
        ),
    )
    return chosen, baseline, sweep


def _validation_report(states, model, threshold, maximum_additions):
    report = {}
    for dataset in dict.fromkeys(state.dataset for state in states):
        dataset_states = [state for state in states if state.dataset == dataset]
        base = _evaluate_threshold(dataset_states, model, 1.0, maximum_additions)
        repaired = _evaluate_threshold(dataset_states, model, threshold, maximum_additions)
        by_type = {}
        for reasoning_type in sorted({str(state.reasoning_type) for state in dataset_states}):
            subset = [state for state in dataset_states if str(state.reasoning_type) == reasoning_type]
            by_type[reasoning_type] = {
                "count": len(subset),
                "baseline": _evaluate_threshold(subset, model, 1.0, maximum_additions),
                "repaired": _evaluate_threshold(subset, model, threshold, maximum_additions),
            }
        report[dataset] = {
            "count": len(dataset_states),
            "baseline": base,
            "repaired": repaired,
            "all_support_gain_points": (repaired["all_support_recall"] - base["all_support_recall"]) * 100.0,
            "by_reasoning_type": by_type,
        }
        print(
            f"{dataset}: all-support={base['all_support_recall']*100:.2f}->"
            f"{repaired['all_support_recall']*100:.2f}, source={base['source_recall']*100:.2f}->"
            f"{repaired['source_recall']*100:.2f}, add={repaired['average_additions']:.2f}, "
            f"repair precision={repaired['repair_precision']*100:.2f}",
            flush=True,
        )
    return report


def main() -> None:
    args = _arguments()
    support_model = joblib.load(args.support_model)["model"]
    width_bundle = joblib.load(args.width_model)
    train_states, training_audit = _build_states(
        args,
        "train",
        Path(args.train_trajectory_root),
        args.max_train_rows_per_dataset,
        support_model,
        width_bundle,
    )
    tuning_model, tuning_fit = _fit(train_states, only_fit=True, seed=args.seed)
    chosen, tuning_baseline, threshold_sweep = _choose_threshold(
        train_states, tuning_model, args.maximum_additions
    )
    print(
        f"chosen threshold={chosen['threshold']:.3f}, tune all-support="
        f"{tuning_baseline['all_support_recall']*100:.2f}->{chosen['all_support_recall']*100:.2f}, "
        f"add={chosen['average_additions']:.2f}, repair precision={chosen['repair_precision']*100:.2f}",
        flush=True,
    )
    # Keep the exact model whose probability scale was calibrated on the
    # disjoint tuning fold.  Refitting on 100% changes HistGB probabilities
    # and invalidates an absolute threshold even when ranking is similar.
    model = tuning_model
    validation_states, validation_audit = _build_states(
        args,
        "validation",
        Path(args.validation_trajectory_root),
        args.max_validation_rows_per_dataset,
        support_model,
        width_bundle,
    )
    validation = _validation_report(
        validation_states, model, chosen["threshold"], args.maximum_additions
    )
    bundle = {
        "format_version": FORMAT_VERSION,
        "method": "local_graph_proof_repair",
        "feature_names": REPAIR_FEATURE_NAMES,
        "model": model,
        "threshold": chosen["threshold"],
        "maximum_additions": args.maximum_additions,
        "neighbors_per_seed": args.neighbors_per_seed,
        "maximum_proposals": args.maximum_proposals,
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report = {
        "format_version": FORMAT_VERSION,
        "method": "local_graph_proof_repair",
        "model": str(model_path),
        "feature_names": list(REPAIR_FEATURE_NAMES),
        "training_audit": training_audit,
        "tuning_fit": tuning_fit,
        "final_fit": {
            **tuning_fit,
            "cohort": "four_of_five_training_rows; fifth_row_per_dataset_used_only_for_threshold",
        },
        "tuning_baseline": tuning_baseline,
        "chosen_threshold": chosen,
        "threshold_sweep": threshold_sweep,
        "validation_audit": validation_audit,
        "validation": validation,
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {model_path}\nreport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
