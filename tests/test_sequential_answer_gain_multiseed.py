import copy

from experiments.analyze_sequential_answer_gain_multiseed import EXPECTED_COUNTS, _aggregate


def _report(seed: int, offset: float) -> dict:
    return {
        "method": "sequential_best_future_f1_gain_hgb",
        "training_root": f"train_pool_seed{seed}_2000",
        "training_sample_seed": seed,
        "training_questions": 6000,
        "training_states": 18000,
        "feature_count": 105,
        "search_cost_f1_fraction": 0.001,
        "validation_root": "full_validation",
        "training_cohort_by_dataset": {
            dataset: {"questions": 2000, "selected_ids_sha256": f"{dataset}-{seed}"}
            for dataset in EXPECTED_COUNTS
        },
        "evaluations": {
            "full_validation": {
                dataset: {
                    "questions": count,
                    "fixed": {"4": {"f1": 50.0, "em": 40.0}},
                    "fixed4_single_reader": {
                        "estimated_latency_seconds": 3.0,
                        "reader_generated_tokens": 50.0,
                    },
                    "policy": {
                        "f1": 51.0 + offset,
                        "em": 41.0 + offset,
                        "average_total_searches": 3.5 + offset / 10,
                        "estimated_latency_seconds": 5.0,
                        "reader_generated_tokens": 100.0,
                    },
                }
                for dataset, count in EXPECTED_COUNTS.items()
            }
        },
    }


def test_aggregate_independent_samples_and_sample_std():
    result = _aggregate([_report(42, -1), _report(43, 0), _report(44, 1)])
    hotpot = result["datasets"]["hotpotqa"]
    assert hotpot["questions"] == 7405
    assert hotpot["across_seeds"]["f1"] == {"mean": 51.0, "sample_std": 1.0}
    assert hotpot["across_seeds"]["delta_f1_points"] == {"mean": 1.0, "sample_std": 1.0}
    assert hotpot["across_seeds"]["average_total_searches"]["mean"] == 3.5


def test_aggregate_rejects_reused_train_sample_and_baseline_change():
    reports = [_report(42, 0), _report(43, 0), _report(44, 0)]
    reports[1]["training_cohort_by_dataset"]["hotpotqa"]["selected_ids_sha256"] = (
        reports[0]["training_cohort_by_dataset"]["hotpotqa"]["selected_ids_sha256"]
    )
    try:
        _aggregate(reports)
    except ValueError as error:
        assert "same training sample" in str(error)
    else:
        raise AssertionError("Reusing the same training sample should fail")
    changed = copy.deepcopy(reports)
    changed[1]["training_cohort_by_dataset"]["hotpotqa"]["selected_ids_sha256"] = "new"
    changed[1]["evaluations"]["full_validation"]["hotpotqa"]["fixed"]["4"]["f1"] += 0.1
    try:
        _aggregate(changed)
    except ValueError as error:
        assert "baseline changed" in str(error)
    else:
        raise AssertionError("Changing the fixed-four baseline should fail")
