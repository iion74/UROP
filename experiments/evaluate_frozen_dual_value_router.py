"""Apply saved five-seed dual-value router to a disjoint cached prefix set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl
from experiments.train_reader_free_student_policy import (
    DECISION_STEP,
    FINAL_STEP,
    policy_feature_names,
    policy_features,
    student_feature_names,
    student_features,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen dual-value ensemble evaluation")
    parser.add_argument("--prefix-root", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--model-root", default="data/models/dual_value_router")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if len(args.seeds) < 3 or len(args.seeds) % 2 == 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds는 중복 없는 3개 이상의 홀수 개여야 합니다.")
    return args


def _routes(evidence_x: np.ndarray, answer_x: np.ndarray, bundles: list[dict]) -> list[tuple[str, dict]]:
    seed_routes = []
    for bundle in bundles:
        evidence = bundle["evidence_model"].predict_proba(evidence_x)[:, 1]
        answer = bundle["answer_model"].predict_proba(answer_x)[:, 1]
        search = evidence >= float(bundle["search_threshold"])
        reason = (~search) & (answer >= float(bundle["answer_threshold"]))
        seed_routes.append(np.where(search, "SEARCH", np.where(reason, "REASON", "STOP")))
    result = []
    majority = len(bundles) // 2 + 1
    for votes in zip(*seed_routes, strict=True):
        counts = Counter(str(vote) for vote in votes)
        if counts["SEARCH"] >= majority:
            route = "SEARCH"
        elif counts["REASON"] > counts["STOP"]:
            route = "REASON"
        else:
            route = "STOP"
        result.append((route, dict(counts)))
    return result


def main() -> None:
    args = _arguments()
    bundles = [joblib.load(Path(args.model_root) / f"policy_seed{seed}.joblib") for seed in args.seeds]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_predictions = []
    all_reason = []
    summaries = {}
    for dataset in tuple(dict.fromkeys(args.datasets)):
        prefix_path = Path(args.prefix_root) / f"{dataset}_{args.split}.jsonl"
        manifest = json.loads(prefix_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "ready":
            raise SystemExit(f"완료되지 않은 prefix: {prefix_path}")
        feature_names = tuple(str(name) for name in manifest["feature_names"])
        for bundle in bundles:
            if tuple(bundle["evidence_feature_names"]) != student_feature_names(feature_names):
                raise SystemExit(f"{dataset}: evidence feature schema 불일치")
            if tuple(bundle["answer_feature_names"]) != policy_feature_names(feature_names, "single_reader"):
                raise SystemExit(f"{dataset}: answer feature schema 불일치")
        prefixes = list(_iter_jsonl(prefix_path))
        if not prefixes:
            raise SystemExit(f"{dataset}: prefix가 없습니다")
        evidence_x = np.vstack([student_features(row, feature_names) for row in prefixes])
        answer_x = np.vstack([policy_features(row, feature_names, "single_reader") for row in prefixes])
        routes = _routes(evidence_x, answer_x, bundles)
        counts = Counter()
        for row, (route, votes) in zip(prefixes, routes, strict=True):
            action = row["actions"][ACTIONS[FINAL_STEP if route == "SEARCH" else DECISION_STEP]]
            counts[route] += 1
            prediction = {
                "id": str(row["id"]),
                "dataset": dataset,
                "route": route,
                "route_votes": votes,
                "prediction_pre_repair": str(action["prediction"]),
                "em_pre_repair": float(action["em"]),
                "f1_pre_repair": float(action["f1"]),
            }
            all_predictions.append(prediction)
            if route == "REASON":
                all_reason.append(
                    {
                        "id": str(row["id"]),
                        "dataset": dataset,
                        "question": str(row["question"]),
                        "route_votes": votes,
                        "baseline_prediction": str(action["prediction"]),
                        "baseline_em": float(action["em"]),
                        "baseline_f1": float(action["f1"]),
                    }
                )
        summaries[dataset] = {"count": len(prefixes), "route_distribution": dict(counts)}
        print(f"{dataset}: frozen routes={dict(counts)}", flush=True)
    predictions_path = output_root / f"frozen_router_{args.split}.predictions.jsonl"
    reason_path = output_root / f"frozen_router_{args.split}.reason_queue.jsonl"
    for path, rows in ((predictions_path, all_predictions), (reason_path, all_reason)):
        with path.open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "format_version": 1,
        "method": "frozen_dual_value_router_five_seed_majority",
        "seeds": args.seeds,
        "model_root": args.model_root,
        "split": args.split,
        "prefix_root": args.prefix_root,
        "predictions": str(predictions_path),
        "reason_queue": str(reason_path),
        "ids_sha256": hashlib.sha256(
            "\n".join(sorted(str(row["id"]) for row in all_predictions)).encode("utf-8")
        ).hexdigest(),
        "datasets": summaries,
    }
    report_path = output_root / f"frozen_router_{args.split}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
