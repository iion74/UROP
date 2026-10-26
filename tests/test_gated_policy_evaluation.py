import unittest

import numpy as np

from experiments.evaluate_gated_policy import (
    aggregate_model_summaries,
    paired_bootstrap_summary,
    predict_one_shot_records,
    summarize_records,
)
from src.policy import POLICY_FEATURE_NAMES


class _FakeGate:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype="float64")

    def predict_graph_probability(self, features, questions):
        return self.probabilities


class _FakeValueModel:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype="float64")

    def predict(self, features):
        return self.predictions


def _row(identifier, dataset, direct, graph):
    return {
        "id": identifier,
        "dataset": dataset,
        "question": "which bridge answer",
        "features": {name: 0.1 for name in POLICY_FEATURE_NAMES},
        "direct_em": direct,
        "direct_f1": direct,
        "oracle_em": max(direct, graph),
        "oracle_f1": max(direct, graph),
        "actions": {
            "vector": {"em": direct, "f1": direct},
            "graph_1hop_small": {"em": graph, "f1": graph},
            "graph_1hop_large": {"em": 0.0, "f1": 0.0},
            "graph_2hop_small": {"em": 0.0, "f1": 0.0},
            "graph_2hop_large": {"em": 0.0, "f1": 0.0},
        },
    }


class GatedPolicyEvaluationTests(unittest.TestCase):
    def test_frozen_gate_routes_without_answer_aware_fallback(self) -> None:
        rows = [_row("hotpotqa:1", "hotpotqa", 1.0, 0.0), _row("hotpotqa:2", "hotpotqa", 0.0, 1.0)]
        payload = {
            "initial_gate": _FakeGate([0.2, 0.8]),
            "one_shot_value_model": _FakeValueModel([[0.9, 0.1, 0.0, -0.1], [0.9, 0.1, 0.0, -0.1]]),
            "one_shot_actions": (
                "graph_1hop_small",
                "graph_1hop_large",
                "graph_2hop_small",
                "graph_2hop_large",
            ),
            "thresholds": {"one_shot": 0.5},
        }

        records = predict_one_shot_records(
            payload,
            rows,
            model_label="test",
            model_seed=42,
            split="validation",
        )

        self.assertEqual([row["selected_action"] for row in records], ["vector", "graph_1hop_small"])
        self.assertEqual([row["policy_f1"] for row in records], [1.0, 1.0])

    def test_summary_is_dataset_macro_not_pooled_micro(self) -> None:
        records = [
            {
                "dataset": "a",
                "selected_action": "vector",
                "direct_em": 0.0,
                "direct_f1": 0.0,
                "static_em": 0.0,
                "static_f1": 0.0,
                "policy_em": 1.0,
                "policy_f1": 1.0,
                "oracle_em": 1.0,
                "oracle_f1": 1.0,
            },
            *[
                {
                    "dataset": "b",
                    "selected_action": "vector",
                    "direct_em": 0.0,
                    "direct_f1": 0.0,
                    "static_em": 0.0,
                    "static_f1": 0.0,
                    "policy_em": 0.0,
                    "policy_f1": 0.0,
                    "oracle_em": 0.0,
                    "oracle_f1": 0.0,
                }
                for _ in range(3)
            ],
        ]

        summary = summarize_records(records, ("a", "b"))

        self.assertAlmostEqual(summary["Macro"]["policy_f1"], 0.5)
        self.assertNotAlmostEqual(summary["Macro"]["policy_f1"], 0.25)

    def test_bootstrap_and_seed_aggregation(self) -> None:
        records = []
        for dataset in ("a", "b"):
            for _ in range(5):
                records.append(
                    {
                        "dataset": dataset,
                        "direct_em": 0.0,
                        "direct_f1": 0.0,
                        "static_em": 0.0,
                        "static_f1": 0.0,
                        "policy_em": 1.0,
                        "policy_f1": 1.0,
                    }
                )
        bootstrap = paired_bootstrap_summary(records, ("a", "b"), samples=100, seed=7)
        self.assertEqual(bootstrap["f1"]["ci95_low"], 1.0)
        self.assertEqual(bootstrap["f1"]["ci95_high"], 1.0)

        models = [
            {"metrics": {"a": {"policy_em": 0.5, "policy_f1": 0.5, "em_gain": 0.1, "f1_gain": 0.1, "policy_minus_static_em": 0.05, "policy_minus_static_f1": 0.05, "graph_rate": 0.2}, "Macro": {"policy_em": 0.5, "policy_f1": 0.5, "em_gain": 0.1, "f1_gain": 0.1, "policy_minus_static_em": 0.05, "policy_minus_static_f1": 0.05, "graph_rate": 0.2}}},
            {"metrics": {"a": {"policy_em": 0.7, "policy_f1": 0.7, "em_gain": 0.3, "f1_gain": 0.3, "policy_minus_static_em": 0.15, "policy_minus_static_f1": 0.15, "graph_rate": 0.4}, "Macro": {"policy_em": 0.7, "policy_f1": 0.7, "em_gain": 0.3, "f1_gain": 0.3, "policy_minus_static_em": 0.15, "policy_minus_static_f1": 0.15, "graph_rate": 0.4}}},
        ]
        aggregate = aggregate_model_summaries(models, ("a",))
        self.assertAlmostEqual(aggregate["Macro"]["f1_gain"]["mean"], 0.2)
        self.assertGreater(aggregate["Macro"]["f1_gain"]["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
