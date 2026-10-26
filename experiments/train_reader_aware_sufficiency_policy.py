"""Train a reader-aware, gold-free-at-inference proof stopping policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl
from src.reader_sufficiency import CHAIN_STRUCTURE_FEATURE_NAMES, chain_structure_features
FORMAT_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reader-aware proof sufficiency policy")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--max-train-rows-per-dataset", type=int, default=1000)
    parser.add_argument(
        "--train-sample-seed",
        type=int,
        help="완료된 train prefix pool에서 데이터셋별 행을 안정적으로 무작위 추출하는 seed",
    )
    parser.add_argument("--additional-train-root")
    parser.add_argument("--additional-train-split", choices=("train", "validation"), default="validation")
    parser.add_argument("--additional-train-offset", type=int, default=0)
    parser.add_argument("--max-additional-train-rows-per-dataset", type=int, default=0)
    parser.add_argument("--max-validation-rows-per-dataset", type=int, default=500)
    parser.add_argument(
        "--validation-offset",
        type=int,
        default=0,
        help="개발에 사용한 validation prefix를 제외하기 위한 행 offset",
    )
    parser.add_argument("--search-cost", type=float, default=0.002)
    parser.add_argument("--f1-weight", type=float, default=0.5)
    parser.add_argument("--em-weight", type=float, default=0.5)
    parser.add_argument("--noninferiority-points", type=float, default=0.25)
    parser.add_argument(
        "--tune-every",
        type=int,
        default=5,
        help="선택된 train cohort에서 매 N번째 문항을 임계값 조정용으로 분리",
    )
    parser.add_argument(
        "--minimum-stop-step",
        type=int,
        choices=(1, 2),
        default=2,
        help="STOP 전에 관찰해야 하는 최소 추가 검색 수; 2는 두 번의 안정성 변화를 요구합니다.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="data/models/reader_aware_sufficiency_policy.joblib")
    parser.add_argument("--report-out", default="data/models/reader_aware_sufficiency_policy.report.json")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_train_rows_per_dataset < 1 or args.max_validation_rows_per_dataset < 0:
        parser.error("train은 1 이상, validation은 0 이상이어야 합니다(0은 전체).")
    if args.tune_every < 2:
        parser.error("tune-every는 2 이상이어야 합니다.")
    if min(args.validation_offset, args.additional_train_offset) < 0:
        parser.error("validation/additional train offset은 0 이상이어야 합니다.")
    if args.max_additional_train_rows_per_dataset < 0:
        parser.error("additional train 행 수는 0 이상이어야 합니다.")
    if bool(args.additional_train_root) != bool(args.max_additional_train_rows_per_dataset):
        parser.error("additional train root와 행 수는 함께 지정해야 합니다.")
    total = args.f1_weight + args.em_weight
    if total <= 0.0 or args.search_cost < 0.0 or args.noninferiority_points < 0.0:
        parser.error("metric weight 합은 양수이고 cost/margin은 0 이상이어야 합니다.")
    args.f1_weight /= total
    args.em_weight /= total
    return args


def _load(
    root: Path,
    datasets: tuple[str, ...],
    split: str,
    maximum: int,
    *,
    offset: int = 0,
    sample_seed: int | None = None,
) -> list[dict]:
    rows = []
    expected_features = None
    for dataset in datasets:
        path = root / f"{dataset}_{split}.jsonl"
        manifest_path = path.with_suffix(".manifest.json")
        if not path.exists() or not manifest_path.exists():
            raise SystemExit(f"reader-aware prefix 결과가 없습니다: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "ready":
            raise SystemExit(f"reader-aware prefix 결과가 완료되지 않았습니다: {manifest_path}")
        feature_names = tuple(manifest.get("feature_names", ()))
        if not feature_names:
            raise SystemExit(f"feature schema가 없습니다: {manifest_path}")
        if expected_features is None:
            expected_features = feature_names
        elif feature_names != expected_features:
            raise SystemExit("데이터셋별 reader-aware feature schema가 다릅니다.")
        available = list(_iter_jsonl(path))[offset:]
        if maximum == 0:
            selected = available
        elif sample_seed is None:
            selected = available[:maximum]
        else:
            selected = sorted(
                available,
                key=lambda row: (
                    hashlib.sha256(
                        f"{sample_seed}\0{dataset}\0{row['id']}".encode("utf-8")
                    ).digest(),
                    str(row["id"]),
                ),
            )[:maximum]
            selected.sort(key=lambda row: str(row["id"]))
        if maximum and len(selected) != maximum:
            raise SystemExit(f"{dataset}: 요청한 {maximum:,}개보다 prefix 행이 적습니다.")
        rows.extend(selected)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("reader-aware prefix 질문 ID가 중복됩니다.")
    return rows


def _utility(action: dict, args) -> float:
    return args.f1_weight * float(action["f1"]) + args.em_weight * float(action["em"])


def _policy_features(row: dict, action: dict) -> list[float]:
    return [
        *action["policy_features"],
        *chain_structure_features(
            row["question"], action.get("evidence_chain", ()), action.get("prediction", "")
        ).tolist(),
    ]


def _state_matrix(rows: list[dict], args):
    features = []
    labels = []
    regrets = []
    metadata = []
    for position, row in enumerate(rows):
        actions = [row["actions"][action] for action in ACTIONS]
        utilities = [_utility(action, args) for action in actions]
        # States before the deployment guard can never emit STOP.  Training on
        # their labels mixes a different decision problem into the only
        # actionable state and made the old classifier unnecessarily brittle.
        for step in range(args.minimum_stop_step, len(ACTIONS) - 1):
            best_future = max(
                utilities[future] - args.search_cost * (future - step)
                for future in range(step + 1, len(ACTIONS))
            )
            regret = best_future - utilities[step]
            features.append(_policy_features(row, actions[step]))
            labels.append(float(regret <= 0.0))
            regrets.append(regret)
            metadata.append((str(row["id"]), str(row["dataset"]), step))
    return (
        np.asarray(features, dtype="float64"),
        np.asarray(labels, dtype="int8"),
        np.asarray(regrets, dtype="float64"),
        metadata,
    )


def _fit(rows: list[dict], args):
    fit_rows = [row for position, row in enumerate(rows) if position % args.tune_every != 0]
    features, labels, regrets, _ = _state_matrix(fit_rows, args)
    weights = 1.0 + 4.0 * np.abs(regrets)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=23,
        min_samples_leaf=35,
        l2_regularization=3.0,
        class_weight="balanced",
        random_state=args.seed,
    )
    regressor = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=23,
        min_samples_leaf=35,
        l2_regularization=3.0,
        random_state=args.seed,
    )
    classifier.fit(features, labels, sample_weight=weights)
    regressor.fit(features, regrets, sample_weight=weights)
    probabilities = classifier.predict_proba(features)[:, 1]
    return classifier, regressor, {
        "states": len(labels),
        "stop_states": int(labels.sum()),
        "stop_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "mean_absolute_regret_error": float(np.mean(np.abs(regressor.predict(features) - regrets))),
    }


def _training_cohort_summary(rows: list[dict], datasets: tuple[str, ...], args) -> dict:
    fit_rows = [row for position, row in enumerate(rows) if position % args.tune_every != 0]
    tuning_rows = [row for position, row in enumerate(rows) if position % args.tune_every == 0]
    result = {}
    for dataset in datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        fit = [row for row in fit_rows if row["dataset"] == dataset]
        tuning = [row for row in tuning_rows if row["dataset"] == dataset]
        _, labels, _, _ = _state_matrix(fit, args)
        selected_ids = "\n".join(sorted(str(row["id"]) for row in selected))
        result[dataset] = {
            "selected_questions": len(selected),
            "fit_questions": len(fit),
            "tuning_questions": len(tuning),
            "fit_states": len(labels),
            "stop_states": int(labels.sum()),
            "continue_states": int(len(labels) - labels.sum()),
            "selected_ids_sha256": hashlib.sha256(selected_ids.encode("utf-8")).hexdigest(),
        }
    return result


def _candidate_parameters() -> list[tuple[float, float, float, float]]:
    result = []
    for probability in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.97, 0.985, 0.995):
        for maximum_regret in (-0.05, -0.02, 0.0, 0.01, 0.02, 0.05):
            for stability in (0.50, 0.80, 0.99):
                for grounding in (0.25, 0.50, 0.75):
                    result.append((probability, maximum_regret, stability, grounding))
    return result


def _feature_indices(feature_names: tuple[str, ...]) -> dict[str, int]:
    required = (
        "answer_is_unknown",
        "chain_is_empty",
        "chain_minimum_document_support",
        "previous_answer_f1",
        "chain_explicit_fact_count",
        "comparison_branches_complete",
        "answer_in_explicit_clause",
        "answer_type_compatible",
    )
    missing = [name for name in required if name not in feature_names]
    if missing:
        raise SystemExit(f"reader-aware feature가 빠졌습니다: {missing}")
    return {name: feature_names.index(name) for name in required}


def _predict_trajectory(
    row: dict,
    classifier,
    regressor,
    parameters: tuple[float, float, float, float],
    indices: dict[str, int],
    minimum_stop_step: int,
    model_output: tuple[np.ndarray, np.ndarray] | None = None,
):
    probability_threshold, maximum_regret, minimum_stability, minimum_grounding = parameters
    actions = [row["actions"][action] for action in ACTIONS]
    features = np.asarray(
        [_policy_features(row, action) for action in actions[:-1]], dtype="float64"
    )
    if model_output is None:
        probabilities = classifier.predict_proba(features)[:, 1]
        predicted_regrets = regressor.predict(features)
    else:
        probabilities, predicted_regrets = model_output
    chosen_step = len(ACTIONS) - 1
    guards = []
    for step, (action, values) in enumerate(zip(actions[:-1], features, strict=True)):
        stable = float(values[indices["previous_answer_f1"]]) >= minimum_stability
        grounded = float(values[indices["chain_minimum_document_support"]]) >= minimum_grounding
        known = float(values[indices["answer_is_unknown"]]) < 0.5
        has_chain = float(values[indices["chain_is_empty"]]) < 0.5
        explicit_fact = float(values[indices["chain_explicit_fact_count"]]) > 0.0
        comparison_complete = float(values[indices["comparison_branches_complete"]]) >= 0.5
        answer_grounded = float(values[indices["answer_in_explicit_clause"]]) >= 0.5
        type_compatible = float(values[indices["answer_type_compatible"]]) >= 0.5
        guard = (
            step >= minimum_stop_step
            and stable
            and grounded
            and known
            and has_chain
            and explicit_fact
            and comparison_complete
            and answer_grounded
            and type_compatible
        )
        guards.append(guard)
        if (
            guard
            and probabilities[step] >= probability_threshold
            and predicted_regrets[step] <= maximum_regret
        ):
            chosen_step = step
            break
    return {
        "chosen_step": chosen_step,
        "selected_action": ACTIONS[chosen_step],
        "stop_probabilities": probabilities.tolist(),
        "predicted_regrets": predicted_regrets.tolist(),
        "safety_guards": guards,
    }


def _model_outputs(rows, classifier, regressor):
    outputs = {}
    for row in rows:
        features = np.asarray(
            [_policy_features(row, row["actions"][action]) for action in ACTIONS[:-1]],
            dtype="float64",
        )
        outputs[str(row["id"])] = (
            classifier.predict_proba(features)[:, 1],
            regressor.predict(features),
        )
    return outputs


def _simulate(
    rows,
    classifier,
    regressor,
    parameters,
    indices,
    minimum_stop_step,
    model_outputs=None,
):
    records = []
    for row in rows:
        prediction = _predict_trajectory(
            row,
            classifier,
            regressor,
            parameters,
            indices,
            minimum_stop_step,
            None if model_outputs is None else model_outputs[str(row["id"])],
        )
        selected = row["actions"][prediction["selected_action"]]
        terminal = row["actions"][ACTIONS[-1]]
        records.append(
            {
                "format_version": FORMAT_VERSION,
                "id": str(row["id"]),
                "dataset": str(row["dataset"]),
                "question": str(row["question"]),
                "reasoning_type": row.get("reasoning_type"),
                "minimum_stop_step": minimum_stop_step,
                **prediction,
                "prediction": selected["prediction"],
                "em": float(selected["em"]),
                "f1": float(selected["f1"]),
                "source_recall": float(selected["source_recall"]),
                "all_support_recall": float(selected["all_support_recall"]),
                "fixed3_prediction": terminal["prediction"],
                "fixed3_em": float(terminal["em"]),
                "fixed3_f1": float(terminal["f1"]),
            }
        )
    return records


def _summary(records: list[dict], datasets: tuple[str, ...]) -> dict:
    result = {}
    for dataset in datasets:
        subset = [row for row in records if row["dataset"] == dataset]
        result[dataset] = {
            "count": len(subset),
            "em": float(np.mean([row["em"] for row in subset])),
            "f1": float(np.mean([row["f1"] for row in subset])),
            "fixed3_em": float(np.mean([row["fixed3_em"] for row in subset])),
            "fixed3_f1": float(np.mean([row["fixed3_f1"] for row in subset])),
            "em_change_points": float(np.mean([row["em"] - row["fixed3_em"] for row in subset]) * 100.0),
            "f1_change_points": float(np.mean([row["f1"] - row["fixed3_f1"] for row in subset]) * 100.0),
            "average_additional_searches": float(np.mean([row["chosen_step"] for row in subset])),
            "step_distribution": dict(sorted(Counter(row["chosen_step"] for row in subset).items())),
        }
    return result


def _choose_parameters(rows, classifier, regressor, feature_names, args):
    tuning_rows = [row for position, row in enumerate(rows) if position % args.tune_every == 0]
    indices = _feature_indices(feature_names)
    model_outputs = _model_outputs(tuning_rows, classifier, regressor)
    sweep = []
    for parameters in _candidate_parameters():
        records = _simulate(
            tuning_rows,
            classifier,
            regressor,
            parameters,
            indices,
            args.minimum_stop_step,
            model_outputs,
        )
        report = _summary(records, args.datasets)
        worst_f1 = min(value["f1_change_points"] for value in report.values())
        worst_em = min(value["em_change_points"] for value in report.values())
        average_search_sum = sum(value["average_additional_searches"] for value in report.values())
        sweep.append(
            {
                "parameters": list(parameters),
                "datasets": report,
                "worst_f1_change_points": worst_f1,
                "worst_em_change_points": worst_em,
                "sum_average_searches": average_search_sum,
            }
        )
    acceptable = [
        row
        for row in sweep
        if row["worst_f1_change_points"] >= -args.noninferiority_points
        and row["worst_em_change_points"] >= -args.noninferiority_points
    ]
    if acceptable:
        chosen = min(
            acceptable,
            key=lambda row: (
                row["sum_average_searches"],
                -row["worst_f1_change_points"],
                -row["worst_em_change_points"],
                tuple(row["parameters"]),
            ),
        )
    else:
        chosen = max(
            sweep,
            key=lambda row: (
                min(row["worst_f1_change_points"], row["worst_em_change_points"]),
                -row["sum_average_searches"],
            ),
        )
    return tuple(float(value) for value in chosen["parameters"]), chosen, sweep


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = _arguments()
    train_rows = _load(
        Path(args.train_root),
        args.datasets,
        "train",
        args.max_train_rows_per_dataset,
        sample_seed=args.train_sample_seed,
    )
    if args.additional_train_root:
        train_rows.extend(
            _load(
                Path(args.additional_train_root),
                args.datasets,
                args.additional_train_split,
                args.max_additional_train_rows_per_dataset,
                offset=args.additional_train_offset,
            )
        )
    validation_rows = _load(
        Path(args.validation_root),
        args.datasets,
        "validation",
        args.max_validation_rows_per_dataset,
        offset=args.validation_offset,
    )
    first_manifest = json.loads(
        (Path(args.train_root) / f"{args.datasets[0]}_train.manifest.json").read_text(encoding="utf-8")
    )
    feature_names = tuple(first_manifest["feature_names"]) + CHAIN_STRUCTURE_FEATURE_NAMES
    classifier, regressor, fit_report = _fit(train_rows, args)
    parameters, chosen, sweep = _choose_parameters(
        train_rows, classifier, regressor, feature_names, args
    )
    indices = _feature_indices(feature_names)
    train_tune = [
        row for position, row in enumerate(train_rows) if position % args.tune_every == 0
    ]
    train_records = _simulate(
        train_tune,
        classifier,
        regressor,
        parameters,
        indices,
        args.minimum_stop_step,
        _model_outputs(train_tune, classifier, regressor),
    )
    validation_records = _simulate(
        validation_rows,
        classifier,
        regressor,
        parameters,
        indices,
        args.minimum_stop_step,
        _model_outputs(validation_rows, classifier, regressor),
    )
    validation = _summary(validation_records, args.datasets)
    for dataset, result in validation.items():
        print(
            f"{dataset}: fixed3/adaptive EM="
            f"{result['fixed3_em']*100:.2f}/{result['em']*100:.2f} F1="
            f"{result['fixed3_f1']*100:.2f}/{result['f1']*100:.2f} "
            f"delta={result['f1_change_points']:+.2f} searches="
            f"{result['average_additional_searches']:.2f}",
            flush=True,
        )
    bundle = {
        "format_version": FORMAT_VERSION,
        "method": "reader_aware_counterfactual_sufficiency",
        "feature_names": feature_names,
        "classifier": classifier,
        "regressor": regressor,
        "parameters": parameters,
        "minimum_stop_step": args.minimum_stop_step,
        "maximum_additional_searches": len(ACTIONS) - 1,
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report_path = Path(args.report_out)
    train_predictions = report_path.with_name(f"{report_path.stem}.train_tune_predictions.jsonl")
    validation_predictions = report_path.with_name(f"{report_path.stem}.validation_predictions.jsonl")
    _write_jsonl(train_predictions, train_records)
    _write_jsonl(validation_predictions, validation_records)
    report = {
        "format_version": FORMAT_VERSION,
        "method": bundle["method"],
        "model": str(model_path),
        "fit": fit_report,
        "training_questions": len(train_rows),
        "training_sample_seed": args.train_sample_seed,
        "tune_every": args.tune_every,
        "fit_questions": len(train_rows) - len(train_tune),
        "tuning_questions": len(train_tune),
        "training_cohort_by_dataset": _training_cohort_summary(
            train_rows, args.datasets, args
        ),
        "minimum_stop_step": args.minimum_stop_step,
        "validation_offset": args.validation_offset,
        "chosen_parameters": chosen,
        "train_tune": _summary(train_records, args.datasets),
        "validation": validation,
        "parameter_candidates": len(sweep),
        "train_tune_predictions": str(train_predictions),
        "validation_predictions": str(validation_predictions),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {model_path}\nreport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
