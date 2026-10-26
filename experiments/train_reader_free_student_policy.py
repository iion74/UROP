"""Train a reader-free student that chooses between search prefixes 2 and 3.

The expensive reader-aware teacher observes a generated answer at every
prefix.  This student only consumes information that already exists after
retrieval (question text, retrieval scores, titles, and candidate turnover).
At deployment the frozen reader is therefore called exactly once, after the
student has selected the final prefix.

Gold answers are used only to construct counterfactual train labels and to
score offline validation.  Dataset identity, gold hops, gold supports, gold
answers, and reader predictions never enter the feature matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl
from experiments.train_reader_aware_sufficiency_policy import _load
from src.answer_gain_policy import READER_HISTORY_FEATURES as _READER_HISTORY_FEATURES
from src.answer_gain_policy import evidence_features as _evidence_features
from src.answer_gain_policy import jaccard as _jaccard
from src.answer_gain_policy import question_features as _question_features
from src.answer_gain_policy import words as _words
from src.reader_sufficiency import CHAIN_STRUCTURE_FEATURE_NAMES, chain_structure_features


FORMAT_VERSION = 1
DECISION_STEP = 2
FINAL_STEP = 3
_TOKENS = re.compile(r"[a-z0-9]+")
_CONTENT_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "both", "by", "did", "do", "does",
        "for", "from", "had", "has", "have", "he", "her", "his", "how", "in", "is", "it", "its",
        "more", "of", "on", "or", "she", "than", "that", "the", "their", "them", "they", "this",
        "to", "was", "were", "what", "when", "where", "which", "who", "with",
    }
)
_CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z0-9'’-]*\b")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reader-free retrieval sufficiency student")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--max-train-rows-per-dataset", type=int, default=1200)
    parser.add_argument("--max-validation-rows-per-dataset", type=int, default=0)
    parser.add_argument("--train-sample-seed", type=int, required=True)
    parser.add_argument("--search-cost", type=float, default=0.002)
    parser.add_argument("--f1-weight", type=float, default=0.5)
    parser.add_argument("--em-weight", type=float, default=0.5)
    parser.add_argument("--noninferiority-points", type=float, default=0.25)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--feature-mode",
        choices=("retrieval_only", "retrieval_graph", "single_reader"),
        default="retrieval_only",
        help="single_reader는 step2 reader를 한 번 검사하고 CONTINUE일 때만 final reader를 추가 호출합니다.",
    )
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_train_rows_per_dataset < 1 or args.max_validation_rows_per_dataset < 0:
        parser.error("train은 1 이상, validation은 0 이상이어야 합니다.")
    if args.cv_folds < 2:
        parser.error("cv-folds는 2 이상이어야 합니다.")
    if args.search_cost < 0.0 or args.noninferiority_points < 0.0:
        parser.error("search cost와 noninferiority margin은 0 이상이어야 합니다.")
    total = args.f1_weight + args.em_weight
    if total <= 0.0:
        parser.error("metric weight의 합은 양수여야 합니다.")
    args.f1_weight /= total
    args.em_weight /= total
    return args


def _content_words(text: str) -> set[str]:
    return _words(text) - _CONTENT_STOPWORDS


def _ordered_content_words(text: str) -> list[str]:
    return [token for token in _TOKENS.findall(str(text).lower()) if token not in _CONTENT_STOPWORDS]


def _entities(text: str) -> set[str]:
    return {
        token.lower().replace("’", "'")
        for token in _CAPITALIZED.findall(str(text))
        if token.lower() not in _CONTENT_STOPWORDS
    }


def _largest_component_fraction(adjacency: list[set[int]]) -> float:
    if not adjacency:
        return 0.0
    unseen = set(range(len(adjacency)))
    largest = 0
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbours = adjacency[node] & unseen
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
        largest = max(largest, size)
    return largest / len(adjacency)


def _content_graph_features(question: str, action: dict, documents: dict[str, dict]) -> tuple[list[str], list[float]]:
    evidence = tuple(action.get("candidate_evidence") or action.get("evidence") or ())
    selected = [
        documents[str(item.get("content_id"))]
        for item in evidence
        if str(item.get("content_id")) in documents
    ]
    question_ordered = _ordered_content_words(question)
    question_terms = set(question_ordered)
    question_entities = _entities(question)
    question_bigrams = set(zip(question_ordered, question_ordered[1:]))
    question_numbers = {token for token in question_terms if token.isdigit()}
    doc_texts = [f"{document.get('title', '')} {document.get('text', '')}" for document in selected]
    doc_terms = [_content_words(text) for text in doc_texts]
    doc_entities = [_entities(text) for text in doc_texts]
    union_terms = set().union(*doc_terms) if doc_terms else set()
    union_entities = set().union(*doc_entities) if doc_entities else set()
    union_ordered = _ordered_content_words(" ".join(doc_texts))
    union_bigrams = set(zip(union_ordered, union_ordered[1:]))
    per_doc_coverage = [len(question_terms & terms) / max(1, len(question_terms)) for terms in doc_terms]
    pair_jaccards = []
    adjacency = [set() for _ in selected]
    bridge_terms: set[str] = set()
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            intersection = doc_terms[left] & doc_terms[right]
            pair_jaccards.append(_jaccard(doc_terms[left], doc_terms[right]))
            shared_entities = doc_entities[left] & doc_entities[right]
            bridge_terms.update(shared_entities)
            if shared_entities or len(intersection) >= 3:
                adjacency[left].add(right)
                adjacency[right].add(left)
    possible_edges = len(selected) * (len(selected) - 1) / 2
    actual_edges = sum(len(neighbours) for neighbours in adjacency) / 2
    cross_mentions = []
    for index, document in enumerate(selected):
        normalized_title = " ".join(_TOKENS.findall(str(document.get("title", "")).lower()))
        other_text = " ".join(
            " ".join(_TOKENS.findall(text.lower()))
            for other_index, text in enumerate(doc_texts)
            if other_index != index
        )
        if normalized_title:
            cross_mentions.append(float(normalized_title in other_text))
    names = [
        "question_content_token_coverage",
        "question_entity_coverage",
        "question_bigram_coverage",
        "maximum_question_document_coverage",
        "mean_question_document_coverage",
        "maximum_document_pair_jaccard",
        "mean_document_pair_jaccard",
        "evidence_graph_edge_density",
        "evidence_graph_largest_component_fraction",
        "title_cross_mention_fraction",
        "bridge_entity_fraction",
        "question_number_coverage",
    ]
    values = [
        len(question_terms & union_terms) / max(1, len(question_terms)),
        len(question_entities & union_entities) / max(1, len(question_entities)),
        len(question_bigrams & union_bigrams) / max(1, len(question_bigrams)),
        max(per_doc_coverage, default=0.0),
        float(np.mean(per_doc_coverage)) if per_doc_coverage else 0.0,
        max(pair_jaccards, default=0.0),
        float(np.mean(pair_jaccards)) if pair_jaccards else 0.0,
        actual_edges / max(1.0, possible_edges),
        _largest_component_fraction(adjacency),
        float(np.mean(cross_mentions)) if cross_mentions else 0.0,
        min(1.0, len(bridge_terms) / 8.0),
        len(question_numbers & union_terms) / max(1, len(question_numbers)),
    ]
    return names, values


def _retrieval_feature_names(manifest_names: tuple[str, ...]) -> tuple[str, ...]:
    try:
        reader_start = manifest_names.index("answer_is_unknown")
    except ValueError as error:
        raise SystemExit("reader feature boundary(answer_is_unknown)를 찾을 수 없습니다.") from error
    names = manifest_names[:reader_start]
    if not names or names[0] != "search_step":
        raise SystemExit("예상한 retrieval feature schema가 아닙니다.")
    return names


def student_feature_names(manifest_names: tuple[str, ...]) -> tuple[str, ...]:
    retrieval_names = _retrieval_feature_names(manifest_names)
    evidence_names, _ = _evidence_features("", {}, set())
    question_names, _ = _question_features("")
    names = []
    for step in range(DECISION_STEP + 1):
        names.extend(f"step{step}_{name}" for name in retrieval_names)
        names.extend(f"step{step}_{name}" for name in evidence_names)
    names.extend(question_names)
    return tuple(names)


def student_features(row: dict, manifest_names: tuple[str, ...]) -> np.ndarray:
    retrieval_count = len(_retrieval_feature_names(manifest_names))
    values: list[float] = []
    previous_ids: set[str] = set()
    for step in range(DECISION_STEP + 1):
        action = row["actions"][ACTIONS[step]]
        policy_features = action.get("policy_features", ())
        if len(policy_features) < retrieval_count:
            raise ValueError(f"{row.get('id')}: retrieval feature가 부족합니다.")
        values.extend(float(value) for value in policy_features[:retrieval_count])
        _, evidence_values = _evidence_features(str(row.get("question", "")), action, previous_ids)
        values.extend(evidence_values)
        evidence = action.get("candidate_evidence") or action.get("evidence") or ()
        previous_ids = {
            str(item.get("content_id", ""))
            for item in evidence
            if item.get("content_id")
        }
    _, question_values = _question_features(str(row.get("question", "")))
    values.extend(question_values)
    result = np.asarray(values, dtype="float64")
    expected = student_feature_names(manifest_names)
    if len(result) != len(expected):
        raise RuntimeError(f"student feature schema mismatch: {len(result)} != {len(expected)}")
    return result


def _single_reader_names(manifest_names: tuple[str, ...]) -> tuple[str, ...]:
    retrieval_count = len(_retrieval_feature_names(manifest_names))
    current_names = tuple(
        name
        for name in manifest_names[retrieval_count:]
        if name not in _READER_HISTORY_FEATURES
    )
    return tuple(f"step2_reader_{name}" for name in current_names) + tuple(
        f"step2_{name}" for name in CHAIN_STRUCTURE_FEATURE_NAMES
    )


def policy_feature_names(manifest_names: tuple[str, ...], mode: str) -> tuple[str, ...]:
    names = student_feature_names(manifest_names)
    if mode == "retrieval_graph":
        content_names, _ = _content_graph_features("", {}, {})
        for step in range(DECISION_STEP + 1):
            names += tuple(f"step{step}_{name}" for name in content_names)
    if mode == "single_reader":
        names += _single_reader_names(manifest_names)
    return names


def policy_features(
    row: dict,
    manifest_names: tuple[str, ...],
    mode: str,
    documents: dict[str, dict] | None = None,
) -> np.ndarray:
    base = student_features(row, manifest_names)
    if mode == "retrieval_only":
        return base
    if mode == "retrieval_graph":
        if documents is None:
            raise ValueError("retrieval_graph mode에는 질문별 원문 document가 필요합니다.")
        content_values = []
        for step in range(DECISION_STEP + 1):
            _, values = _content_graph_features(
                str(row.get("question", "")), row["actions"][ACTIONS[step]], documents
            )
            content_values.extend(values)
        result = np.concatenate((base, np.asarray(content_values, dtype="float64")))
        expected = policy_feature_names(manifest_names, mode)
        if len(result) != len(expected):
            raise RuntimeError(f"graph feature schema mismatch: {len(result)} != {len(expected)}")
        return result
    if mode != "single_reader":
        raise ValueError(f"지원하지 않는 feature mode: {mode}")
    retrieval_count = len(_retrieval_feature_names(manifest_names))
    action = row["actions"][ACTIONS[DECISION_STEP]]
    current = [
        float(value)
        for name, value in zip(manifest_names[retrieval_count:], action["policy_features"][retrieval_count:], strict=True)
        if name not in _READER_HISTORY_FEATURES
    ]
    structure = chain_structure_features(
        str(row.get("question", "")),
        action.get("evidence_chain", ()),
        str(action.get("prediction", "")),
    )
    result = np.concatenate((base, np.asarray(current, dtype="float64"), structure))
    expected = policy_feature_names(manifest_names, mode)
    if len(result) != len(expected):
        raise RuntimeError(f"policy feature schema mismatch: {len(result)} != {len(expected)}")
    return result


def _utility(action: dict, args) -> float:
    return args.f1_weight * float(action["f1"]) + args.em_weight * float(action["em"])


def _matrix(rows: list[dict], manifest_names: tuple[str, ...], args, document_rows=None):
    features = np.vstack(
        [
            policy_features(
                row,
                manifest_names,
                args.feature_mode,
                None if document_rows is None else document_rows[str(row["id"])],
            )
            for row in rows
        ]
    )
    step2 = np.asarray([_utility(row["actions"][ACTIONS[DECISION_STEP]], args) for row in rows])
    step3 = np.asarray([_utility(row["actions"][ACTIONS[FINAL_STEP]], args) for row in rows])
    gains = step3 - args.search_cost - step2
    labels = (gains > 0.0).astype("int8")
    datasets = np.asarray([str(row["dataset"]) for row in rows])
    return features, labels, gains, datasets


MODEL_SPECS = {
    "hgb_absolute": {
        "learning_rate": 0.04,
        "max_iter": 300,
        "max_leaf_nodes": 23,
        "min_samples_leaf": 35,
        "l2_regularization": 3.0,
        "regressor_loss": "absolute_error",
        "family": "hist_gradient_boosting",
    },
    "hgb_squared": {
        "learning_rate": 0.035,
        "max_iter": 350,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 5.0,
        "regressor_loss": "squared_error",
        "family": "hist_gradient_boosting",
    },
    "hgb_quantile80": {
        "learning_rate": 0.03,
        "max_iter": 350,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 35,
        "l2_regularization": 5.0,
        "regressor_loss": "quantile",
        "quantile": 0.80,
        "family": "hist_gradient_boosting",
    },
    "hgb_quantile90": {
        "learning_rate": 0.03,
        "max_iter": 350,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 35,
        "l2_regularization": 5.0,
        "regressor_loss": "quantile",
        "quantile": 0.90,
        "family": "hist_gradient_boosting",
    },
    "hgb_quantile95": {
        "learning_rate": 0.03,
        "max_iter": 350,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 35,
        "l2_regularization": 5.0,
        "regressor_loss": "quantile",
        "quantile": 0.95,
        "family": "hist_gradient_boosting",
    },
    "extra_trees": {
        "family": "extra_trees",
        "n_estimators": 300,
        "max_features": 0.7,
        "min_samples_leaf": 4,
        "max_depth": None,
    },
}


def _fit_pair(features, labels, gains, seed: int, spec_name: str):
    configured = MODEL_SPECS[spec_name]
    family = configured["family"]
    weights = 1.0 + 12.0 * np.abs(gains) + 30.0 * np.maximum(gains, 0.0)
    if family == "extra_trees":
        tree_options = {
            key: value
            for key, value in configured.items()
            if key != "family"
        }
        classifier = ExtraTreesClassifier(
            **tree_options,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        regressor = ExtraTreesRegressor(
            **tree_options,
            random_state=seed,
            n_jobs=-1,
        )
        classifier.fit(features, labels, sample_weight=weights)
        regressor.fit(features, gains, sample_weight=weights)
        return classifier, regressor
    spec = {
        key: value
        for key, value in configured.items()
        if key not in {"family", "regressor_loss", "quantile"}
    }
    classifier = HistGradientBoostingClassifier(
        **spec,
        class_weight="balanced",
        random_state=seed,
    )
    regressor_options = {
        **spec,
        "loss": configured["regressor_loss"],
        "random_state": seed,
    }
    if configured.get("regressor_loss") == "quantile":
        regressor_options["quantile"] = configured["quantile"]
    regressor = HistGradientBoostingRegressor(**regressor_options)
    classifier.fit(features, labels, sample_weight=weights)
    regressor.fit(features, gains, sample_weight=weights)
    return classifier, regressor


def _oof_predictions(features, labels, gains, datasets, seed: int, folds: int, spec_name: str):
    strata = np.asarray([f"{dataset}:{label}" for dataset, label in zip(datasets, labels, strict=True)])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.zeros(len(labels), dtype="float64")
    predicted_gains = np.zeros(len(labels), dtype="float64")
    for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(features, strata)):
        classifier, regressor = _fit_pair(
            features[fit_indices], labels[fit_indices], gains[fit_indices], seed + fold, spec_name
        )
        probabilities[holdout_indices] = classifier.predict_proba(features[holdout_indices])[:, 1]
        predicted_gains[holdout_indices] = regressor.predict(features[holdout_indices])
    return probabilities, predicted_gains


def _candidate_parameters():
    probabilities = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.995, 1.01)
    gains = (-0.05, -0.02, -0.01, -0.005, 0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 1.0)
    return [(probability, gain) for probability in probabilities for gain in gains]


def _policy_continue(probabilities, predicted_gains, parameters):
    probability_threshold, gain_threshold = parameters
    return (probabilities >= probability_threshold) | (predicted_gains >= gain_threshold)


def _score_policy(rows, continue_mask, datasets):
    result = {}
    for dataset in datasets:
        indices = np.asarray([row["dataset"] == dataset for row in rows])
        subset_rows = [row for row, include in zip(rows, indices, strict=True) if include]
        subset_continue = continue_mask[indices]
        adaptive_em = np.asarray(
            [
                float(row["actions"][ACTIONS[FINAL_STEP if should_continue else DECISION_STEP]]["em"])
                for row, should_continue in zip(subset_rows, subset_continue, strict=True)
            ]
        )
        adaptive_f1 = np.asarray(
            [
                float(row["actions"][ACTIONS[FINAL_STEP if should_continue else DECISION_STEP]]["f1"])
                for row, should_continue in zip(subset_rows, subset_continue, strict=True)
            ]
        )
        fixed_em = np.asarray([float(row["actions"][ACTIONS[FINAL_STEP]]["em"]) for row in subset_rows])
        fixed_f1 = np.asarray([float(row["actions"][ACTIONS[FINAL_STEP]]["f1"]) for row in subset_rows])
        stop_rate = float(np.mean(~subset_continue))
        result[dataset] = {
            "count": len(subset_rows),
            "em": float(adaptive_em.mean()),
            "f1": float(adaptive_f1.mean()),
            "fixed3_em": float(fixed_em.mean()),
            "fixed3_f1": float(fixed_f1.mean()),
            "em_change_points": float((adaptive_em - fixed_em).mean() * 100.0),
            "f1_change_points": float((adaptive_f1 - fixed_f1).mean() * 100.0),
            "average_additional_searches": float(3.0 - stop_rate),
            "search_reduction_percent_vs_fixed3": float(stop_rate / 3.0 * 100.0),
            "step_distribution": {"2": int((~subset_continue).sum()), "3": int(subset_continue.sum())},
        }
    return result


def _choose(rows, datasets, model_outputs, margin_points: float):
    sweep = []
    for spec_name, (probabilities, predicted_gains) in model_outputs.items():
        for parameters in _candidate_parameters():
            continue_mask = _policy_continue(probabilities, predicted_gains, parameters)
            report = _score_policy(rows, continue_mask, datasets)
            worst_em = min(item["em_change_points"] for item in report.values())
            worst_f1 = min(item["f1_change_points"] for item in report.values())
            minimum_reduction = min(item["search_reduction_percent_vs_fixed3"] for item in report.values())
            reduction_sum = sum(item["search_reduction_percent_vs_fixed3"] for item in report.values())
            sweep.append(
                {
                    "model_spec": spec_name,
                    "parameters": list(parameters),
                    "datasets": report,
                    "worst_em_change_points": worst_em,
                    "worst_f1_change_points": worst_f1,
                    "minimum_search_reduction_percent": minimum_reduction,
                    "sum_search_reduction_percent": reduction_sum,
                }
            )
    selected_tier = -margin_points
    acceptable = [
        item
        for item in sweep
        if item["worst_em_change_points"] >= selected_tier
        and item["worst_f1_change_points"] >= selected_tier
    ]
    if not acceptable:
        acceptable = sweep
        selected_tier = None
    chosen = max(
        acceptable,
        key=lambda item: (
            item["minimum_search_reduction_percent"],
            item["sum_search_reduction_percent"],
            item["worst_f1_change_points"],
            item["worst_em_change_points"],
            item["model_spec"],
            tuple(item["parameters"]),
        ),
    )
    return chosen, sweep, selected_tier


def _prediction_records(rows, probabilities, predicted_gains, parameters, method: str):
    continue_mask = _policy_continue(probabilities, predicted_gains, parameters)
    records = []
    for row, probability, gain, should_continue in zip(
        rows, probabilities, predicted_gains, continue_mask, strict=True
    ):
        chosen_step = FINAL_STEP if should_continue else DECISION_STEP
        selected = row["actions"][ACTIONS[chosen_step]]
        fixed = row["actions"][ACTIONS[FINAL_STEP]]
        records.append(
            {
                "format_version": FORMAT_VERSION,
                "method": method,
                "id": str(row["id"]),
                "dataset": str(row["dataset"]),
                "chosen_step": chosen_step,
                "selected_action": ACTIONS[chosen_step],
                "continue_probability": float(probability),
                "predicted_marginal_gain": float(gain),
                "prediction": selected["prediction"],
                "em": float(selected["em"]),
                "f1": float(selected["f1"]),
                "source_recall": float(selected["source_recall"]),
                "all_support_recall": float(selected["all_support_recall"]),
                "fixed3_prediction": fixed["prediction"],
                "fixed3_em": float(fixed["em"]),
                "fixed3_f1": float(fixed["f1"]),
            }
        )
    return records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _load_manifest_names(root: Path, dataset: str) -> tuple[str, ...]:
    manifest = json.loads((root / f"{dataset}_train.manifest.json").read_text(encoding="utf-8"))
    names = tuple(str(name) for name in manifest.get("feature_names", ()))
    if not names:
        raise SystemExit("train manifest에 feature_names가 없습니다.")
    return names


def _load_document_rows(
    root: Path,
    datasets: tuple[str, ...],
    split: str,
    required_ids: set[str],
) -> dict[str, dict[str, dict]]:
    result = {}
    for dataset in datasets:
        path = root / dataset / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"processed document가 없습니다: {path}")
        for row in _iter_jsonl(path):
            row_id = f"{dataset}:{row.get('id')}"
            if row_id not in required_ids:
                continue
            result[row_id] = {
                str(document["content_id"]): {
                    "title": str(document.get("title", "")),
                    "text": str(document.get("text", "")),
                }
                for document in row.get("documents", ())
                if document.get("content_id")
            }
    missing = required_ids - result.keys()
    if missing:
        raise SystemExit(f"processed document mapping이 {len(missing):,}개 부족합니다.")
    return result


def main() -> None:
    args = _arguments()
    train_root = Path(args.train_root)
    validation_root = Path(args.validation_root)
    train_rows = _load(
        train_root,
        args.datasets,
        "train",
        args.max_train_rows_per_dataset,
        sample_seed=args.train_sample_seed,
    )
    validation_rows = _load(
        validation_root,
        args.datasets,
        "validation",
        args.max_validation_rows_per_dataset,
    )
    manifest_names = _load_manifest_names(train_root, args.datasets[0])
    names = policy_feature_names(manifest_names, args.feature_mode)
    train_documents = validation_documents = None
    if args.feature_mode == "retrieval_graph":
        processed_root = Path(args.processed_root)
        train_documents = _load_document_rows(
            processed_root,
            args.datasets,
            "train",
            {str(row["id"]) for row in train_rows},
        )
        validation_documents = _load_document_rows(
            processed_root,
            args.datasets,
            "validation",
            {str(row["id"]) for row in validation_rows},
        )
    features, labels, gains, datasets = _matrix(
        train_rows, manifest_names, args, train_documents
    )

    model_outputs = {}
    diagnostics = {}
    for spec_name in MODEL_SPECS:
        probabilities, predicted_gains = _oof_predictions(
            features, labels, gains, datasets, args.seed, args.cv_folds, spec_name
        )
        model_outputs[spec_name] = (probabilities, predicted_gains)
        diagnostics[spec_name] = {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "mean_absolute_gain_error": float(np.mean(np.abs(predicted_gains - gains))),
        }

    chosen, sweep, selected_tier = _choose(
        train_rows, args.datasets, model_outputs, args.noninferiority_points
    )
    spec_name = str(chosen["model_spec"])
    parameters = tuple(float(value) for value in chosen["parameters"])
    classifier, regressor = _fit_pair(features, labels, gains, args.seed, spec_name)
    validation_features, _, _, _ = _matrix(
        validation_rows, manifest_names, args, validation_documents
    )
    validation_probabilities = classifier.predict_proba(validation_features)[:, 1]
    validation_gains = regressor.predict(validation_features)
    method = (
        "reader_free_retrieval_student"
        if args.feature_mode == "retrieval_only"
        else (
            "reader_free_evidence_graph_student"
            if args.feature_mode == "retrieval_graph"
            else "single_reader_selective_student"
        )
    )
    records = _prediction_records(
        validation_rows, validation_probabilities, validation_gains, parameters, method
    )
    validation_summary = _score_policy(
        validation_rows,
        _policy_continue(validation_probabilities, validation_gains, parameters),
        args.datasets,
    )
    for dataset, result in validation_summary.items():
        print(
            f"{dataset}: fixed3/student EM={result['fixed3_em']*100:.2f}/{result['em']*100:.2f} "
            f"F1={result['fixed3_f1']*100:.2f}/{result['f1']*100:.2f} "
            f"delta={result['f1_change_points']:+.2f} searches={result['average_additional_searches']:.2f}",
            flush=True,
        )

    bundle = {
        "format_version": FORMAT_VERSION,
        "method": method,
        "feature_names": names,
        "source_manifest_feature_names": manifest_names,
        "decision_step": DECISION_STEP,
        "maximum_additional_searches": FINAL_STEP,
        "model_spec": spec_name,
        "classifier": classifier,
        "regressor": regressor,
        "parameters": parameters,
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report_path = Path(args.report_out)
    predictions_path = report_path.with_name(f"{report_path.stem}.validation_predictions.jsonl")
    _write_jsonl(predictions_path, records)
    selected_ids = "\n".join(sorted(str(row["id"]) for row in train_rows))
    label_counts = Counter(int(value) for value in labels)
    report = {
        "format_version": FORMAT_VERSION,
        "method": bundle["method"],
        "model": str(model_path),
        "validation_predictions": str(predictions_path),
        "training_sample_seed": args.train_sample_seed,
        "training_questions": len(train_rows),
        "validation_questions": len(validation_rows),
        "feature_count": len(names),
        "feature_names": names,
        "forbidden_feature_audit": {
            "uses_dataset_identity": False,
            "uses_gold_hop": False,
            "uses_gold_answer": False,
            "uses_gold_support": False,
            "uses_reader_prediction": args.feature_mode == "single_reader",
        },
        "training_ids_sha256": hashlib.sha256(selected_ids.encode("utf-8")).hexdigest(),
        "continue_states": int(label_counts[1]),
        "stop_states": int(label_counts[0]),
        "continue_rate": float(labels.mean()),
        "cross_validation_folds": args.cv_folds,
        "model_diagnostics": diagnostics,
        "selected_noninferiority_tier_points": selected_tier,
        "chosen": chosen,
        "parameter_candidates": len(sweep),
        "validation": validation_summary,
        "cost_accounting": {
            "reader_calls": (
                "exactly one final reader call at the selected prefix"
                if args.feature_mode != "single_reader"
                else "one verifier/final call at step2, plus a final call only when continuing to step3"
            ),
            "reader_generated_tokens": (
                "only the selected prefix generation"
                if args.feature_mode != "single_reader"
                else "step2 generation plus step3 generation only when continuing"
            ),
            "planner": "query-rewrite calls up to the selected prefix remain",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {model_path}\nreport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
