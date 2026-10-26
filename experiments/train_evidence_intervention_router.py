"""Route between Direct RAG and proof-repaired evidence before generation.

The development prefix supplies paired answer utilities, but inference
features contain only the question and retrieval/proof state.  Dataset ID,
gold hop, reasoning type, gold answer, EM, and F1 are never model inputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl


FORMAT_VERSION = 1
TOKEN = re.compile(r"[A-Za-z0-9]+")
NUMERIC_FEATURE_NAMES = (
    "question_token_count",
    "question_character_count_log",
    "question_capitalized_span_count",
    "question_possessive_count",
    "question_of_count",
    "question_relative_count",
    "question_conjunction_count",
    "question_comma_count",
    "question_comparison_terms",
    "question_boolean",
    "selected_document_count",
    "selected_probability_max",
    "selected_probability_mean",
    "selected_probability_min",
    "selected_probability_gap",
    "repair_count",
    "repair_syndrome_probability",
    "question_title_anchor_count",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual evidence intervention router")
    parser.add_argument(
        "--method-root",
        default="data/benchmarks/multihop/causal_evidence_reader/factorized_graph_repair_validation_2000",
    )
    parser.add_argument(
        "--topology-root",
        default="data/benchmarks/multihop/causal_evidence_reader/topology3_adaptive_validation_2000",
    )
    parser.add_argument("--model-out", default="data/models/evidence_intervention_router.joblib")
    parser.add_argument("--report-out", default="data/models/evidence_intervention_router.report.json")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--development-prefix", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(args.development_prefix, args.folds, args.bootstrap_samples) < 1:
        parser.error("development/folds/bootstrap 값은 1 이상이어야 합니다.")
    return args


def _numeric(row: dict, topology: dict) -> list[float]:
    question = str(row["question"])
    tokens = TOKEN.findall(question.lower())
    lowered = " ".join(tokens)
    probabilities = [float(value) for value in topology.get("selected_probabilities", ())]
    sorted_probabilities = sorted(probabilities, reverse=True)
    titles = [str(item.get("title", "")).lower() for item in row.get("evidence", ())]
    question_lower = question.lower()
    anchor_count = sum(len(title) >= 4 and title in question_lower for title in set(titles))
    return [
        float(len(tokens)),
        float(np.log1p(len(question))),
        float(len(re.findall(r"(?:\b[A-Z][\w'.-]*\b(?:\s+|$))+", question))),
        float(question.count("'s")),
        float(tokens.count("of")),
        float(sum(tokens.count(word) for word in ("that", "where", "who", "whose", "which"))),
        float(sum(tokens.count(word) for word in ("and", "or", "both"))),
        float(question.count(",")),
        float(sum(tokens.count(word) for word in ("earlier", "later", "more", "less", "same", "older", "younger", "both"))),
        float(lowered.startswith(("are ", "do ", "does ", "did ", "is ", "was ", "were "))),
        float(len(row.get("base_selected_indices", ()))),
        max(probabilities, default=0.0),
        float(np.mean(probabilities)) if probabilities else 0.0,
        min(probabilities, default=0.0),
        (sorted_probabilities[0] - sorted_probabilities[1]) if len(sorted_probabilities) > 1 else 0.0,
        float(row.get("repair_count", 0)),
        float(row.get("syndrome_probability") or 0.0),
        float(anchor_count),
    ]


def _load(args):
    rows = []
    for dataset in args.datasets:
        method = list(_iter_jsonl(Path(args.method_root) / f"{dataset}_{args.split}.jsonl"))
        topology = {
            str(row["id"]): row
            for row in _iter_jsonl(Path(args.topology_root) / f"{dataset}_{args.split}.jsonl")
        }
        for position, row in enumerate(method):
            source = topology[str(row["id"])]
            rows.append(
                {
                    "id": str(row["id"]),
                    "dataset": dataset,
                    "position": position,
                    "question": str(row["question"]),
                    "numeric": _numeric(row, source),
                    "direct_em": float(source["baseline_em"]),
                    "direct_f1": float(source["baseline_f1"]),
                    "method_em": float(row["em"]),
                    "method_f1": float(row["f1"]),
                }
            )
    return rows


def _fit(rows, seed):
    questions = [row["question"] for row in rows]
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=2_000,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=3,
        max_features=2_000,
        sublinear_tf=True,
    )
    scaler = StandardScaler()
    numeric = scaler.fit_transform(np.asarray([row["numeric"] for row in rows], dtype="float64"))
    matrix = hstack((word.fit_transform(questions), char.fit_transform(questions), csr_matrix(numeric)))
    labels = np.asarray([float(row["method_f1"] > row["direct_f1"]) for row in rows], dtype="int8")
    weights = np.asarray(
        [0.25 + abs(row["method_f1"] - row["direct_f1"]) for row in rows], dtype="float64"
    )
    model = LogisticRegression(
        C=0.20,
        max_iter=2_000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(matrix, labels, sample_weight=weights)
    return {"word": word, "char": char, "scaler": scaler, "model": model}


def _predict(bundle, rows):
    questions = [row["question"] for row in rows]
    numeric = bundle["scaler"].transform(
        np.asarray([row["numeric"] for row in rows], dtype="float64")
    )
    matrix = hstack(
        (
            bundle["word"].transform(questions),
            bundle["char"].transform(questions),
            csr_matrix(numeric),
        )
    )
    return bundle["model"].predict_proba(matrix)[:, 1]


def _paired_interval(differences, samples, seed):
    differences = np.asarray(differences, dtype="float64")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype="float64")
    for start in range(0, samples, 1_000):
        size = min(1_000, samples - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        draws[start : start + size] = differences[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "gain": float(differences.mean() * 100.0),
        "ci95_low": float(low * 100.0),
        "ci95_high": float(high * 100.0),
        "probability_positive": float(np.mean(draws > 0.0) * 100.0),
    }


def _summarize(rows, probabilities, threshold, bootstrap_samples, seed):
    report = {}
    for offset, dataset in enumerate(dict.fromkeys(row["dataset"] for row in rows)):
        indices = [index for index, row in enumerate(rows) if row["dataset"] == dataset]
        selected = [probabilities[index] >= threshold for index in indices]
        routed_em = [
            rows[index]["method_em"] if use_method else rows[index]["direct_em"]
            for index, use_method in zip(indices, selected, strict=True)
        ]
        routed_f1 = [
            rows[index]["method_f1"] if use_method else rows[index]["direct_f1"]
            for index, use_method in zip(indices, selected, strict=True)
        ]
        direct_em = [rows[index]["direct_em"] for index in indices]
        direct_f1 = [rows[index]["direct_f1"] for index in indices]
        method_em = [rows[index]["method_em"] for index in indices]
        method_f1 = [rows[index]["method_f1"] for index in indices]
        report[dataset] = {
            "count": len(indices),
            "method_route_rate": float(np.mean(selected) * 100.0),
            "direct_em": float(np.mean(direct_em) * 100.0),
            "method_em": float(np.mean(method_em) * 100.0),
            "routed_em": float(np.mean(routed_em) * 100.0),
            "direct_f1": float(np.mean(direct_f1) * 100.0),
            "method_f1": float(np.mean(method_f1) * 100.0),
            "routed_f1": float(np.mean(routed_f1) * 100.0),
            "paired_bootstrap_vs_direct": {
                "em": _paired_interval(
                    np.asarray(routed_em) - np.asarray(direct_em), bootstrap_samples, seed + offset * 2
                ),
                "f1": _paired_interval(
                    np.asarray(routed_f1) - np.asarray(direct_f1), bootstrap_samples, seed + offset * 2 + 1
                ),
            },
        }
    return report


def _choose_oof_threshold(development, folds, seed):
    decisive = [
        row for row in development if abs(row["method_f1"] - row["direct_f1"]) > 1e-12
    ]
    labels = np.asarray([row["method_f1"] > row["direct_f1"] for row in decisive], dtype="int8")
    probabilities = np.zeros(len(decisive), dtype="float64")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_indices, test_indices in splitter.split(np.zeros(len(decisive)), labels):
        bundle = _fit([decisive[index] for index in train_indices], seed)
        probabilities[test_indices] = _predict(bundle, [decisive[index] for index in test_indices])
    candidates = np.linspace(0.20, 0.80, 61)
    sweep = []
    for threshold in candidates:
        gains = {}
        rates = {}
        for dataset in dict.fromkeys(row["dataset"] for row in development):
            indices = [index for index, row in enumerate(decisive) if row["dataset"] == dataset]
            chosen = probabilities[indices] >= threshold
            differences = [
                decisive[index]["method_f1"] - decisive[index]["direct_f1"]
                if use_method
                else 0.0
                for index, use_method in zip(indices, chosen, strict=True)
            ]
            # Report gain on the full development cohort, including zero-difference ties.
            full_count = sum(row["dataset"] == dataset for row in development)
            gains[dataset] = float(sum(differences) * 100.0 / full_count)
            rates[dataset] = float(np.mean(chosen)) if len(chosen) else 0.0
        sweep.append(
            {
                "threshold": float(threshold),
                "gains": gains,
                "decisive_method_route_rates": rates,
                "minimum_dataset_gain": min(gains.values()),
                "sum_dataset_gains": sum(gains.values()),
            }
        )
    chosen = max(
        sweep,
        key=lambda row: (
            row["minimum_dataset_gain"],
            row["sum_dataset_gains"],
            row["threshold"],
        ),
    )
    return chosen, sweep, len(decisive)


def main() -> None:
    args = _arguments()
    rows = _load(args)
    development = [row for row in rows if row["position"] < args.development_prefix]
    heldout = [row for row in rows if row["position"] >= args.development_prefix]
    chosen, sweep, decisive_count = _choose_oof_threshold(
        development, args.folds, args.seed
    )
    decisive_development = [
        row for row in development if abs(row["method_f1"] - row["direct_f1"]) > 1e-12
    ]
    bundle = _fit(decisive_development, args.seed)
    heldout_probabilities = _predict(bundle, heldout)
    heldout_report = _summarize(
        heldout,
        heldout_probabilities,
        chosen["threshold"],
        args.bootstrap_samples,
        args.seed,
    )
    artifact = {
        "format_version": FORMAT_VERSION,
        "method": "counterfactual_evidence_intervention_router",
        "feature_policy": "question_text_and_observable_retrieval_state_only",
        "numeric_feature_names": NUMERIC_FEATURE_NAMES,
        "development_prefix": args.development_prefix,
        "threshold": chosen["threshold"],
        **bundle,
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    report = {
        "format_version": FORMAT_VERSION,
        "method": artifact["method"],
        "model": str(model_path),
        "development_rows": len(development),
        "decisive_development_rows": decisive_count,
        "heldout_rows": len(heldout),
        "chosen_oof_threshold": chosen,
        "threshold_sweep": sweep,
        "heldout": heldout_report,
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OOF threshold={chosen['threshold']:.3f}, development gains={chosen['gains']}", flush=True)
    for dataset, result in heldout_report.items():
        interval = result["paired_bootstrap_vs_direct"]["f1"]
        print(
            f"{dataset}: route={result['method_route_rate']:.2f}% direct/method/routed EM="
            f"{result['direct_em']:.2f}/{result['method_em']:.2f}/{result['routed_em']:.2f} F1="
            f"{result['direct_f1']:.2f}/{result['method_f1']:.2f}/{result['routed_f1']:.2f} "
            f"gain={interval['gain']:+.2f} CI=[{interval['ci95_low']:+.2f},{interval['ci95_high']:+.2f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
