"""Evaluate fixed-depth and equal-search-budget baselines on cached prefixes.

The random baseline is deliberately pre-repair: random SEARCH uses the step-3
answer and all other cases use the step-2 answer.  It is compared only with the
router's pre-repair answer, never with the full system's REASON/judge outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analyze_context_reader_aware_multiseed import _paired_interval
from experiments.analyze_student_policy_multiseed import _planner_elapsed
from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KCI fixed2 and matched-budget random baselines")
    parser.add_argument(
        "--prefix-root",
        default="data/benchmarks/multihop/context_reader_aware/full_validation",
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument(
        "--all-reason-judge-root",
        default="",
        help="모든 질문의 thinking+judge 반사실 결과가 있으면 동일 검색/추론 예산 무작위 3-way 정책을 비교합니다.",
    )
    parser.add_argument(
        "--full-predictions",
        default=(
            "data/benchmarks/multihop/context_reader_aware/analysis/"
            "dual_value_full_system.predictions.jsonl"
        ),
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--random-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=(
            "data/benchmarks/multihop/context_reader_aware/analysis/"
            "kci_required_baselines.json"
        ),
    )
    args = parser.parse_args()
    if min(args.bootstrap_samples, args.random_draws) < 1:
        parser.error("bootstrap samples와 random draws는 1 이상이어야 합니다.")
    return args


def _vectors(
    prefixes: list[dict], predictions: list[dict], all_reason_judged: dict[str, dict] | None = None
) -> dict[str, np.ndarray]:
    if len(prefixes) != len(predictions):
        raise ValueError("prefix와 final prediction의 행 수가 다릅니다.")
    by_id = {str(row["id"]): row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("final prediction ID가 중복되었습니다.")
    prefix_ids = {str(row["id"]) for row in prefixes}
    if prefix_ids != set(by_id):
        raise ValueError("prefix와 final prediction의 ID가 다릅니다.")
    outputs: dict[str, list[float]] = {
        name: []
        for name in (
            "direct_em", "direct_f1", "fixed2_em", "fixed2_f1", "fixed3_em",
            "fixed3_f1", "fixed2_source_recall", "fixed3_source_recall",
            "fixed2_latency_seconds", "fixed3_latency_seconds", "fixed2_reader_tokens",
            "fixed3_reader_tokens", "router_em", "router_f1", "router_source_recall",
            "router_search", "router_reason", "final_em", "final_f1", "final_search",
            "final_latency_seconds", "final_reader_tokens", "final_reader_calls",
            "all_reason_em", "all_reason_f1",
        )
    }
    for prefix in prefixes:
        final = by_id[str(prefix["id"])]
        actions = prefix["actions"]
        direct, second, third = (actions[ACTIONS[step]] for step in (0, 2, 3))
        route = str(final["route"])
        if route not in {"SEARCH", "REASON", "STOP"}:
            raise ValueError(f"알 수 없는 route: {route}")
        router = third if route == "SEARCH" else second
        initial = float(prefix.get("initial_retrieval_latency_seconds") or 0.0)
        for name, action in (("direct", direct), ("fixed2", second), ("fixed3", third), ("router", router)):
            for metric in ("em", "f1"):
                outputs[f"{name}_{metric}"].append(float(action[metric]))
        for name, action in (("fixed2", second), ("fixed3", third)):
            outputs[f"{name}_source_recall"].append(float(action["source_recall"]))
            outputs[f"{name}_reader_tokens"].append(float(action.get("reader_generated_tokens") or 0.0))
            outputs[f"{name}_latency_seconds"].append(
                initial + _planner_elapsed(prefix, 2 if name == "fixed2" else 3)
                + float(action.get("reader_latency_seconds") or 0.0)
            )
        outputs["router_source_recall"].append(float(router["source_recall"]))
        outputs["router_search"].append(float(route == "SEARCH"))
        outputs["router_reason"].append(float(route == "REASON"))
        outputs["final_search"].append(float(final["searches"]))
        outputs["final_em"].append(float(final["em"]))
        outputs["final_f1"].append(float(final["f1"]))
        if all_reason_judged is not None:
            judged = all_reason_judged[str(prefix["id"])]
            for metric in ("em", "f1"):
                judged_value = float(judged[f"judged_{metric}"])
                outputs[f"all_reason_{metric}"].append(judged_value)
                if route == "REASON" and abs(judged_value - float(final[metric])) > 1e-9:
                    raise ValueError(f"REASON 반사실 결과가 기존 최종 결과와 다릅니다: {prefix['id']}")
        for output_name, field in (
            ("final_latency_seconds", "latency_seconds"),
            ("final_reader_tokens", "reader_generated_tokens"),
            ("final_reader_calls", "reader_calls"),
        ):
            if field in final:
                outputs[output_name].append(float(final[field]))
        if int(final["searches"]) != (3 if route == "SEARCH" else 2):
            raise ValueError(f"route/searches 불일치: {prefix['id']}")
    return {key: np.asarray(values, dtype="float64") for key, values in outputs.items()}


def _random_three_way(
    vectors: dict[str, np.ndarray], draws: int, bootstrap_samples: int, seed: int
) -> dict:
    """Randomize all routes at exact SEARCH and REASON counts using all potential outcomes."""
    rng = np.random.default_rng(seed)
    count = len(vectors["final_f1"])
    search_count = int(vectors["router_search"].sum())
    reason_count = int(vectors["router_reason"].sum())
    result = {
        "search_count": search_count,
        "reason_count": reason_count,
        "stop_count": count - search_count - reason_count,
        "average_searches": 2.0 + search_count / count,
        "average_reader_calls": 1.0 + 2.0 * reason_count / count,
        "random_draws": draws,
    }
    for index, metric in enumerate(("em", "f1")):
        second = vectors[f"fixed2_{metric}"]
        search_delta = vectors[f"fixed3_{metric}"] - second
        reason_delta = vectors[f"all_reason_{metric}"] - second
        if len(reason_delta) != count:
            raise ValueError("all-reason 반사실 결과에 누락된 질문이 있습니다.")
        expected = second + (search_count * search_delta + reason_count * reason_delta) / count
        final = vectors[f"final_{metric}"]
        sampled = np.empty(draws, dtype="float64")
        base = float(second.mean())
        for draw in range(draws):
            shuffled = rng.permutation(count)
            search_indices = shuffled[:search_count]
            reason_indices = shuffled[search_count : search_count + reason_count]
            sampled[draw] = (
                base
                + (search_delta[search_indices].sum() + reason_delta[reason_indices].sum()) / count
            )
        result[metric] = {
            "full_system": float(final.mean()),
            "random_expected": float(expected.mean()),
            "full_minus_random_expected": _paired_interval(
                final - expected, bootstrap_samples, seed + index + 1
            ),
            "random_draw_95pct_low": float(np.quantile(sampled, 0.025)),
            "random_draw_95pct_high": float(np.quantile(sampled, 0.975)),
            "fraction_random_at_least_full": float(np.mean(sampled >= final.mean())),
        }
    return result


def _random_matched_budget(
    vectors: dict[str, np.ndarray], draws: int, bootstrap_samples: int, seed: int
) -> dict:
    """Uniformly choose exactly the router's SEARCH count in every draw."""
    rng = np.random.default_rng(seed)
    search_mask = vectors["router_search"] > 0.5
    count = len(search_mask)
    search_count = int(search_mask.sum())
    rate = search_count / count
    step2_f1 = vectors["fixed2_f1"]
    step3_f1 = vectors["fixed3_f1"]
    step2_em = vectors["fixed2_em"]
    step3_em = vectors["fixed3_em"]
    step2_support = vectors["fixed2_source_recall"]
    step3_support = vectors["fixed3_source_recall"]
    f1_delta = step3_f1 - step2_f1
    em_delta = step3_em - step2_em
    support_delta = step3_support - step2_support
    sampled_f1 = np.empty(draws, dtype="float64")
    sampled_em = np.empty(draws, dtype="float64")
    sampled_support = np.empty(draws, dtype="float64")
    for index in range(draws):
        chosen = rng.choice(count, size=search_count, replace=False)
        sampled_f1[index] = step2_f1.mean() + f1_delta[chosen].sum() / count
        sampled_em[index] = step2_em.mean() + em_delta[chosen].sum() / count
        sampled_support[index] = step2_support.mean() + support_delta[chosen].sum() / count

    result = {"search_count": search_count, "average_searches": 2.0 + rate, "random_draws": draws}
    for metric, sampled, step2, delta in (
        ("f1", sampled_f1, step2_f1, f1_delta),
        ("em", sampled_em, step2_em, em_delta),
        ("source_recall", sampled_support, step2_support, support_delta),
    ):
        observed = vectors[f"router_{metric}"]
        # This expectation is exact for uniform sampling without replacement.
        expected_per_question = step2 + rate * delta
        result[metric] = {
            "router": float(observed.mean()),
            "random_expected": float(expected_per_question.mean()),
            "router_minus_random_expected": _paired_interval(
                observed - expected_per_question, bootstrap_samples, seed + {"f1": 1, "em": 2, "source_recall": 3}[metric]
            ),
            "random_draw_mean": float(sampled.mean()),
            "random_draw_95pct_low": float(np.quantile(sampled, 0.025)),
            "random_draw_95pct_high": float(np.quantile(sampled, 0.975)),
            "fraction_random_at_least_router": float(np.mean(sampled >= observed.mean())),
        }
    return result


def _random_search_given_reason(
    vectors: dict[str, np.ndarray], draws: int, bootstrap_samples: int, seed: int
) -> dict:
    """Randomize SEARCH only, holding the actual REASON queue and judge fixed."""
    rng = np.random.default_rng(seed)
    reason = vectors["router_reason"] > 0.5
    eligible = np.flatnonzero(~reason)
    count = len(reason)
    search_count = int(vectors["router_search"].sum())
    if search_count > len(eligible):
        raise ValueError("SEARCH count가 REASON 외 문항보다 많습니다.")
    rate = search_count / len(eligible)
    result = {
        "reason_queue_count": int(reason.sum()),
        "search_count": search_count,
        "average_searches": 2.0 + search_count / count,
        "random_draws": draws,
    }
    for index, metric in enumerate(("em", "f1")):
        second = vectors[f"fixed2_{metric}"]
        delta = vectors[f"fixed3_{metric}"] - second
        final = vectors[f"final_{metric}"]
        expected = np.where(reason, final, second + rate * delta)
        constant = float(np.where(reason, final, second).sum() / count)
        sampled = np.empty(draws, dtype="float64")
        for draw in range(draws):
            chosen = rng.choice(eligible, size=search_count, replace=False)
            sampled[draw] = constant + delta[chosen].sum() / count
        result[metric] = {
            "full_system": float(final.mean()),
            "random_search_expected": float(expected.mean()),
            "full_minus_random_expected": _paired_interval(
                final - expected, bootstrap_samples, seed + index + 1
            ),
            "random_draw_95pct_low": float(np.quantile(sampled, 0.025)),
            "random_draw_95pct_high": float(np.quantile(sampled, 0.975)),
            "fraction_random_at_least_full": float(np.mean(sampled >= final.mean())),
        }
    return result


def _summarize(vectors: dict[str, np.ndarray], draws: int, bootstrap_samples: int, seed: int) -> dict:
    result = {"count": len(vectors["final_f1"]), "methods": {}, "comparisons": {}}
    for name, searches in (("direct", 0.0), ("fixed2", 2.0), ("fixed3", 3.0), ("router_pre_repair", 2.0 + vectors["router_search"].mean()), ("full_system", vectors["final_search"].mean())):
        key = "router" if name == "router_pre_repair" else "final" if name == "full_system" else name
        result["methods"][name] = {
            "em": float(vectors[f"{key}_em"].mean()),
            "f1": float(vectors[f"{key}_f1"].mean()),
            "average_searches": float(searches),
        }
        if key in {"fixed2", "fixed3"}:
            result["methods"][name].update(
                {
                    "source_recall": float(vectors[f"{key}_source_recall"].mean()),
                    "estimated_latency_seconds": float(vectors[f"{key}_latency_seconds"].mean()),
                    "reader_generated_tokens": float(vectors[f"{key}_reader_tokens"].mean()),
                }
            )
        if key == "final" and len(vectors["final_latency_seconds"]) == len(vectors["final_f1"]):
            result["methods"][name].update(
                {
                    "estimated_latency_seconds": float(vectors["final_latency_seconds"].mean()),
                    "reader_generated_tokens": float(vectors["final_reader_tokens"].mean()),
                    "reader_calls": float(vectors["final_reader_calls"].mean()),
                }
            )
    for index, (left, right) in enumerate(
        (("final", "fixed2"), ("final", "fixed3"), ("fixed3", "fixed2"))
    ):
        result["comparisons"][f"{left}_vs_{right}"] = {
            metric: _paired_interval(
                vectors[f"{left}_{metric}"] - vectors[f"{right}_{metric}"],
                bootstrap_samples,
                seed + 10 + index * 2 + offset,
            )
            for offset, metric in enumerate(("em", "f1"))
        }
    result["matched_budget_random_pre_repair"] = _random_matched_budget(
        vectors, draws, bootstrap_samples, seed + 100
    )
    result["matched_budget_random_search_given_reason"] = _random_search_given_reason(
        vectors, draws, bootstrap_samples, seed + 200
    )
    if len(vectors["all_reason_f1"]) == len(vectors["final_f1"]):
        result["matched_budget_random_three_way"] = _random_three_way(
            vectors, draws, bootstrap_samples, seed + 300
        )
    return result


def main() -> None:
    args = _arguments()
    datasets = tuple(dict.fromkeys(args.datasets))
    predictions_by_dataset: dict[str, list[dict]] = {dataset: [] for dataset in datasets}
    for row in _iter_jsonl(Path(args.full_predictions)):
        dataset = str(row["dataset"])
        if dataset in predictions_by_dataset:
            predictions_by_dataset[dataset].append(row)
    results = {}
    for index, dataset in enumerate(datasets):
        prefixes = list(_iter_jsonl(Path(args.prefix_root) / f"{dataset}_{args.split}.jsonl"))
        judged = None
        if args.all_reason_judge_root:
            judged_path = Path(args.all_reason_judge_root) / f"{dataset}_{args.split}.jsonl"
            judged_rows = list(_iter_jsonl(judged_path))
            judged = {str(row["id"]): row for row in judged_rows}
            if len(judged) != len(judged_rows) or set(judged) != {str(row["id"]) for row in prefixes}:
                raise ValueError(f"{dataset}: all-reason judged 결과 ID coverage가 불완전합니다.")
        vectors = _vectors(prefixes, predictions_by_dataset[dataset], judged)
        results[dataset] = _summarize(
            vectors, args.random_draws, args.bootstrap_samples, args.seed + index * 1000
        )
        fixed2 = results[dataset]["methods"]["fixed2"]
        final = results[dataset]["methods"]["full_system"]
        random_result = results[dataset]["matched_budget_random_pre_repair"]["f1"]
        print(
            f"{dataset}: n={len(prefixes):,}, FIXED2 F1={fixed2['f1']*100:.2f}, "
            f"final F1={final['f1']*100:.2f}, "
            f"router-random F1={random_result['router_minus_random_expected']['observed_change_points']:+.2f}p",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "format_version": 1,
        "scope": (
            "complete_official_question_local_validation"
            if args.split == "validation"
            else "sampled_disjoint_train_holdout"
        ),
        "split": args.split,
        "prefix_root": args.prefix_root,
        "full_predictions": args.full_predictions,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "random_draws": args.random_draws,
        "latency_note": "FIXED2/3 latency is estimated from cached prefix planner and final-reader timing, not an end-to-end replay.",
        "random_note": "Random and router are compared before REASON/judge; each random draw has exactly the same SEARCH count as the router on that dataset.",
        "conditional_random_note": "The second randomization holds the actual REASON queue and judged answers fixed and randomizes SEARCH only among remaining questions at the exact same search budget. It isolates search targeting, not the whole policy.",
        "all_reason_judge_root": args.all_reason_judge_root,
        "three_way_random_note": "When all-reason counterfactual outputs are supplied, SEARCH/REASON/STOP are uniformly randomized at the policy's exact per-dataset counts; both search and extra reader-call budgets match the final system.",
        "datasets": results,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
