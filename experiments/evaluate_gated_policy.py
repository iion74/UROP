"""Evaluate frozen precision-gated one-shot retrieval policies.

This command never fits or calibrates a model.  It loads Answer Oracle rows for
an unseen split, lets each frozen policy choose one of the already evaluated
retrieval actions from question/evidence features alone, and only then reads
the selected action's EM/F1.  Multiple policy seeds can be evaluated in one
command without another reader call.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from experiments.evaluate_answer_oracle import DIRECT_ACTION, OUTPUT_FORMAT_VERSION
from experiments.train_answer_policy import ONE_SHOT_GRAPH_ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from src.policy import POLICY_FEATURE_NAMES


EVALUATION_FORMAT_VERSION = 1
POLICY_TYPE = "question_evidence_gated_answer_policy"
# Legacy pilot fallback.  Final full-data models store the static action chosen
# from train Macro F1 in their payload, before validation is inspected.
FROZEN_STATIC_ACTION = "graph_1hop_large"


def main() -> None:
    args = _arguments()
    started = perf_counter()
    root = Path(args.rows_root)
    rows, answer_signature = _load_rows(
        root,
        args.datasets,
        args.split,
        args.max_rows_per_dataset,
    )
    if args.split == "validation":
        _assert_no_train_overlap(rows, root, args.datasets)

    _log(
        f"동결 Gated one-shot 정책 평가 시작: split={args.split} · "
        f"{len(rows):,}문항 · 모델 {len(args.models)}개"
    )
    _log("정책 입력은 질문+검색 특징뿐이며 validation 정답은 행동 선택 후 EM/F1 계산에만 사용합니다.")

    all_records: list[dict] = []
    model_summaries: list[dict] = []
    for model_index, model_name in enumerate(args.models):
        model_path = Path(model_name)
        payload, model_metadata = _load_model(model_path)
        training_signature = model_metadata.get("answer_oracle_signature")
        if training_signature is not None and training_signature != answer_signature:
            raise SystemExit(
                f"학습과 평가의 reader/검색 signature가 다릅니다: {model_path}"
            )
        records = predict_one_shot_records(
            payload,
            rows,
            model_label=model_path.stem,
            model_seed=model_metadata.get("seed"),
            split=args.split,
        )
        summary = summarize_records(records, args.datasets)
        bootstrap = {
            "policy_vs_direct": paired_bootstrap_summary(
                records,
                args.datasets,
                baseline="direct",
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + model_index * 10,
            ),
            "policy_vs_frozen_static": paired_bootstrap_summary(
                records,
                args.datasets,
                baseline="static",
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + model_index * 10 + 2,
            ),
        }
        result = {
            "model": str(model_path),
            "model_label": model_path.stem,
            "seed": model_metadata.get("seed"),
            "threshold": float(payload["thresholds"]["one_shot"]),
            "frozen_static_action": payload.get("frozen_static_action", FROZEN_STATIC_ACTION),
            "metrics": summary,
            "paired_bootstrap": bootstrap,
        }
        model_summaries.append(result)
        all_records.extend(records)
        _report_model(result, args.datasets)

    seed_summary = aggregate_model_summaries(model_summaries, args.datasets)
    _report_seed_summary(seed_summary, args.datasets)
    predictions_path = Path(args.predictions_out)
    summary_path = (
        Path(args.summary_out)
        if args.summary_out
        else predictions_path.with_suffix(".summary.json")
    )
    _write_predictions(predictions_path, all_records)
    _write_summary(
        summary_path,
        {
            "format_version": EVALUATION_FORMAT_VERSION,
            "split": args.split,
            "datasets": list(args.datasets),
            "rows": len(rows),
            "answer_oracle_signature": answer_signature,
            "models": model_summaries,
            "across_policy_seeds": seed_summary,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
    )
    _log(f"\n문항별 예측 저장: {predictions_path}")
    _log(f"평가 요약 저장: {summary_path}")
    _log(f"동결 정책 평가 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="동결 Gated 검색 정책의 unseen split EM/F1 평가")
    parser.add_argument("--rows-root", default="data/benchmarks/multihop/answer_oracle")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "data/models/gated_answer_policy_1000.joblib",
            "data/models/gated_answer_policy_1000_seed43.joblib",
            "data/models/gated_answer_policy_1000_seed44.joblib",
        ],
    )
    parser.add_argument(
        "--predictions-out",
        default="data/benchmarks/multihop/policy_eval/gated_validation_predictions.jsonl",
    )
    parser.add_argument("--summary-out")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    args.models = tuple(dict.fromkeys(args.models))
    if args.max_rows_per_dataset < 0 or args.bootstrap_samples < 0:
        parser.error("max rows와 bootstrap samples는 0 이상이어야 합니다.")
    if not args.models:
        parser.error("평가할 정책 모델이 하나 이상 필요합니다.")
    return args


def _load_rows(
    root: Path,
    datasets: tuple[str, ...],
    split: str,
    limit: int,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    signatures: list[dict] = []
    for dataset in datasets:
        path = root / f"{dataset}_{split}.jsonl"
        manifest_path = path.with_suffix(".manifest.json")
        if not path.exists() or not manifest_path.exists():
            raise SystemExit(f"Answer Oracle 결과 또는 manifest가 없습니다: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_answer_manifest(manifest, dataset, split, manifest_path)
        dataset_rows = list(_iter_jsonl(path))
        if limit:
            dataset_rows = dataset_rows[:limit]
        if not dataset_rows:
            raise SystemExit(f"평가할 Answer Oracle 행이 없습니다: {path}")
        for row in dataset_rows:
            _validate_row(row, dataset, path)
        rows.extend(dataset_rows)
        signatures.append(_answer_signature(manifest))

    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("평가 행에 중복 ID가 있습니다.")
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise SystemExit("데이터셋별 Answer Oracle의 reader/retrieval 설정이 서로 다릅니다.")
    return rows, signatures[0]


def _validate_answer_manifest(
    manifest: Mapping,
    dataset: str,
    split: str,
    path: Path,
) -> None:
    actions = set(manifest.get("actions", ()))
    required_actions = {DIRECT_ACTION, *ONE_SHOT_GRAPH_ACTIONS}
    if (
        int(manifest.get("format_version", -1)) != OUTPUT_FORMAT_VERSION
        or manifest.get("dataset") != dataset
        or manifest.get("split") != split
        or manifest.get("status") != "ready"
        or not required_actions.issubset(actions)
    ):
        raise SystemExit(f"완료된 공정 비교용 Answer Oracle manifest가 아닙니다: {path}")


def _answer_signature(manifest: Mapping) -> dict:
    reader = manifest.get("reader", {})
    signature = {
        key: manifest.get(key)
        for key in (
            "context_mode",
            "top_k",
            "initial_candidates",
            "context_character_budget",
            "corpus_seed",
            "query_seed",
            "query_sampling_version",
            "actions",
            "chunk_size",
            "chunk_overlap",
            "embedding_model",
            "reranker_model",
            "reranker_candidates",
            "graph_format_version",
            "graph_semantic_neighbors",
            "graph_title_bridges",
            "graph_max_title_bridges_per_chunk",
            "router_seed_count",
            "graph_small_budget",
            "graph_large_budget",
            "graph_expansion_decay",
            "graph_rerank_prior_weight",
            "graph_source_diversity_bonus",
            "graph_min_expanded_results",
        )
    }
    signature["actions"] = list(manifest.get("actions", ()))
    signature["reader"] = {
        key: reader.get(key)
        for key in (
            "model",
            "model_digest",
            "max_tokens",
            "seed",
            "temperature",
            "prompt_version",
        )
    }
    return signature


def _validate_row(row: Mapping, dataset: str, path: Path) -> None:
    required_actions = {DIRECT_ACTION, *ONE_SHOT_GRAPH_ACTIONS}
    if (
        int(row.get("format_version", -1)) != OUTPUT_FORMAT_VERSION
        or row.get("dataset") != dataset
        or not required_actions.issubset(row.get("actions", {}))
        or any(name not in row.get("features", {}) for name in POLICY_FEATURE_NAMES)
    ):
        raise SystemExit(f"지원하지 않는 Answer Oracle 행입니다: {path} · {row.get('id')}")


def _assert_no_train_overlap(rows: list[dict], root: Path, datasets: Iterable[str]) -> None:
    validation_ids = {str(row["id"]) for row in rows}
    train_ids: set[str] = set()
    checked: list[str] = []
    for dataset in datasets:
        candidates = (
            root / f"{dataset}_train_sequential_v6.jsonl",
            root / f"{dataset}_train.jsonl",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            continue
        train_ids.update(str(row["id"]) for row in _iter_jsonl(path))
        checked.append(dataset)
    overlap = validation_ids & train_ids
    if overlap:
        examples = ", ".join(sorted(overlap)[:3])
        raise SystemExit(f"train-validation ID 누출이 발견됐습니다: {examples}")
    if checked:
        _log(f"train-validation ID 중복 검사 통과: {len(validation_ids):,} validation · 중복 0")
    else:
        _log("주의: train 행 파일이 없어 ID 중복 검사를 생략했습니다.")


def _load_model(path: Path) -> tuple[dict, dict]:
    if not path.exists():
        raise SystemExit(f"정책 모델이 없습니다: {path}")
    payload = joblib.load(path)
    required = {
        "policy_type",
        "feature_names",
        "one_shot_actions",
        "thresholds",
        "initial_gate",
        "one_shot_value_model",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise SystemExit(f"지원하지 않는 정책 모델입니다: {path}")
    if (
        payload["policy_type"] != POLICY_TYPE
        or tuple(payload["feature_names"]) != tuple(POLICY_FEATURE_NAMES)
        or tuple(payload["one_shot_actions"]) != tuple(ONE_SHOT_GRAPH_ACTIONS)
    ):
        raise SystemExit(f"정책 feature/action 형식이 현재 evaluator와 다릅니다: {path}")
    threshold = float(payload["thresholds"].get("one_shot", np.nan))
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SystemExit(f"유효한 고정 one-shot threshold가 없습니다: {path}")
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise SystemExit(f"학습 설정 manifest가 없습니다: {manifest_path}")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("policy_type") != POLICY_TYPE or metadata.get("datasets") != list(
        SUPPORTED_DATASETS
    ):
        raise SystemExit(f"세 데이터셋으로 동결한 Gated 정책 manifest가 아닙니다: {manifest_path}")
    return payload, metadata


def predict_one_shot_records(
    payload: Mapping,
    rows: list[dict],
    *,
    model_label: str,
    model_seed: int | None,
    split: str,
) -> list[dict]:
    """Apply a frozen gate/value model without using answer labels for routing."""

    features = np.asarray(
        [[float(row["features"][name]) for name in POLICY_FEATURE_NAMES] for row in rows],
        dtype="float64",
    )
    questions = [str(row["question"]) for row in rows]
    probabilities = payload["initial_gate"].predict_graph_probability(features, questions)
    predicted_values = np.asarray(payload["one_shot_value_model"].predict(features))
    actions = tuple(payload["one_shot_actions"])
    static_action = str(payload.get("frozen_static_action", FROZEN_STATIC_ACTION))
    if static_action not in actions:
        raise ValueError(f"동결 static action이 Graph action space에 없습니다: {static_action}")
    threshold = float(payload["thresholds"]["one_shot"])
    if predicted_values.shape != (len(rows), len(actions)):
        raise ValueError("정책 value model의 출력 차원이 action space와 다릅니다.")

    records: list[dict] = []
    for row, probability, estimates in zip(rows, probabilities, predicted_values, strict=True):
        action_index = int(np.argmax(estimates))
        selected_action = actions[action_index] if float(probability) > threshold else DIRECT_ACTION
        selected = row["actions"][selected_action]
        records.append(
            {
                "format_version": EVALUATION_FORMAT_VERSION,
                "id": str(row["id"]),
                "dataset": str(row["dataset"]),
                "split": split,
                "model_label": model_label,
                "model_seed": model_seed,
                "gate_probability": float(probability),
                "gate_threshold": threshold,
                "predicted_graph_action": actions[action_index],
                "predicted_graph_advantage": float(estimates[action_index]),
                "selected_action": selected_action,
                "direct_em": float(row["direct_em"]),
                "direct_f1": float(row["direct_f1"]),
                "static_action": static_action,
                "static_em": float(row["actions"][static_action]["em"]),
                "static_f1": float(row["actions"][static_action]["f1"]),
                "policy_em": float(selected["em"]),
                "policy_f1": float(selected["f1"]),
                "oracle_em": float(row["oracle_em"]),
                "oracle_f1": float(row["oracle_f1"]),
            }
        )
    return records


def summarize_records(records: list[dict], datasets: Iterable[str]) -> dict:
    summary: dict[str, dict] = {}
    datasets = tuple(datasets)
    for name in (*datasets, "Macro"):
        selected = records if name == "Macro" else [row for row in records if row["dataset"] == name]
        if name == "Macro":
            values = {
                metric: float(
                    np.mean(
                        [
                            np.mean([row[metric] for row in records if row["dataset"] == dataset])
                            for dataset in datasets
                        ]
                    )
                )
                for metric in (
                    "direct_em",
                    "direct_f1",
                    "static_em",
                    "static_f1",
                    "policy_em",
                    "policy_f1",
                    "oracle_em",
                    "oracle_f1",
                )
            }
        else:
            values = {
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in (
                    "direct_em",
                    "direct_f1",
                    "static_em",
                    "static_f1",
                    "policy_em",
                    "policy_f1",
                    "oracle_em",
                    "oracle_f1",
                )
            }
        potential = values["oracle_f1"] - values["direct_f1"]
        summary[name] = {
            "n": len(selected),
            **values,
            "em_gain": values["policy_em"] - values["direct_em"],
            "f1_gain": values["policy_f1"] - values["direct_f1"],
            "static_em_gain": values["static_em"] - values["direct_em"],
            "static_f1_gain": values["static_f1"] - values["direct_f1"],
            "policy_minus_static_em": values["policy_em"] - values["static_em"],
            "policy_minus_static_f1": values["policy_f1"] - values["static_f1"],
            "oracle_potential_f1": potential,
            "oracle_potential_capture": (
                (values["policy_f1"] - values["direct_f1"]) / potential
                if potential > 1e-12
                else 0.0
            ),
            "graph_rate": float(
                np.mean([row["selected_action"] != DIRECT_ACTION for row in selected])
            ),
            "actions": dict(sorted(Counter(row["selected_action"] for row in selected).items())),
        }
    return summary


def paired_bootstrap_summary(
    records: list[dict],
    datasets: Iterable[str],
    *,
    baseline: str = "direct",
    samples: int,
    seed: int,
) -> dict:
    if samples == 0:
        return {}
    datasets = tuple(datasets)
    result: dict[str, dict] = {}
    for metric in ("em", "f1"):
        differences = {
            dataset: np.asarray(
                [
                    row[f"policy_{metric}"] - row[f"{baseline}_{metric}"]
                    for row in records
                    if row["dataset"] == dataset
                ],
                dtype="float64",
            )
            for dataset in datasets
        }
        draws = _stratified_macro_draws(differences, samples=samples, seed=seed + (metric == "f1"))
        result[metric] = {
            "observed_gain": float(np.mean([values.mean() for values in differences.values()])),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "probability_positive": float((np.count_nonzero(draws > 0.0) + 1) / (samples + 1)),
        }
    return result


def _stratified_macro_draws(
    differences: Mapping[str, np.ndarray],
    *,
    samples: int,
    seed: int,
    chunk_size: int = 500,
) -> np.ndarray:
    if not differences or any(len(values) == 0 for values in differences.values()):
        raise ValueError("bootstrap에는 모든 데이터셋의 paired difference가 필요합니다.")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype="float64")
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        chunk = np.zeros(stop - start, dtype="float64")
        for values in differences.values():
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            chunk += values[indices].mean(axis=1)
        draws[start:stop] = chunk / len(differences)
    return draws


def aggregate_model_summaries(model_summaries: list[dict], datasets: Iterable[str]) -> dict:
    result: dict[str, dict] = {}
    for name in (*tuple(datasets), "Macro"):
        result[name] = {}
        for metric in (
            "policy_em",
            "policy_f1",
            "em_gain",
            "f1_gain",
            "policy_minus_static_em",
            "policy_minus_static_f1",
            "graph_rate",
        ):
            values = np.asarray(
                [model["metrics"][name][metric] for model in model_summaries],
                dtype="float64",
            )
            result[name][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return result


def _report_model(result: dict, datasets: Iterable[str]) -> None:
    _log(
        f"\n[{result['model_label']}] seed={result['seed']} · "
        f"고정 gate threshold={result['threshold']:.3f}"
    )
    for name in (*tuple(datasets), "Macro"):
        metric = result["metrics"][name]
        _log(
            f"{name}: n={metric['n']:,} | Direct EM/F1={metric['direct_em']:.3f}/{metric['direct_f1']:.3f} "
            f"| Static({result['frozen_static_action']})="
            f"{metric['static_em']:.3f}/{metric['static_f1']:.3f} "
            f"| Policy EM/F1={metric['policy_em']:.3f}/{metric['policy_f1']:.3f} "
            f"| gain={metric['em_gain']:+.3f}/{metric['f1_gain']:+.3f} "
            f"| Policy−Static ΔF1={metric['policy_minus_static_f1']:+.3f} "
            f"| Oracle F1={metric['oracle_f1']:.3f} | capture={metric['oracle_potential_capture']:.1%} "
            f"| Graph 선택={metric['graph_rate']:.1%}"
        )
    bootstrap = result["paired_bootstrap"]
    if bootstrap["policy_vs_direct"]:
        direct_em = bootstrap["policy_vs_direct"]["em"]
        direct_f1 = bootstrap["policy_vs_direct"]["f1"]
        static_f1 = bootstrap["policy_vs_frozen_static"]["f1"]
        _log(
            "Macro paired bootstrap 95% CI: "
            f"Policy−Direct ΔEM [{direct_em['ci95_low']:+.4f},{direct_em['ci95_high']:+.4f}] · "
            f"ΔF1 [{direct_f1['ci95_low']:+.4f},{direct_f1['ci95_high']:+.4f}] "
            f"· P>0={direct_f1['probability_positive']:.4f}"
        )
        _log(
            f"Policy−Static({result['frozen_static_action']}) ΔF1 95% CI: "
            f"[{static_f1['ci95_low']:+.4f},{static_f1['ci95_high']:+.4f}] "
            f"· P>0={static_f1['probability_positive']:.4f}"
        )
    _log(f"전체 행동 분포: {result['metrics']['Macro']['actions']}")


def _report_seed_summary(summary: dict, datasets: Iterable[str]) -> None:
    _log("\n[정책 seed 평균±표준편차]")
    for name in (*tuple(datasets), "Macro"):
        metric = summary[name]
        _log(
            f"{name}: Policy EM={metric['policy_em']['mean']:.3f}±{metric['policy_em']['std']:.3f} "
            f"| F1={metric['policy_f1']['mean']:.3f}±{metric['policy_f1']['std']:.3f} "
            f"| ΔF1={metric['f1_gain']['mean']:+.3f}±{metric['f1_gain']['std']:.3f} "
            f"| vs Static-Graph={metric['policy_minus_static_f1']['mean']:+.3f}"
            f"±{metric['policy_minus_static_f1']['std']:.3f} "
            f"| Graph={metric['graph_rate']['mean']:.1%}±{metric['graph_rate']['std']:.1%}"
        )


def _write_predictions(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for record in sorted(records, key=lambda item: (item["model_label"], item["dataset"], item["id"])):
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
