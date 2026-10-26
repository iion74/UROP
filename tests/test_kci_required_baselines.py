import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.analyze_kci_required_baselines import (
    _random_matched_budget,
    _random_three_way,
    _vectors,
)
from experiments.evaluate_frozen_dual_value_router import _routes
from experiments.analyze_dual_value_full_system import _seed_predictions
from experiments.train_adaptive_proof_completion_policy import ACTIONS


class _Probabilities:
    def __init__(self, positive):
        self.positive = np.asarray(positive, dtype="float64")

    def predict_proba(self, features):
        assert len(features) == len(self.positive)
        return np.column_stack((1.0 - self.positive, self.positive))


class KciBaselineTests(unittest.TestCase):
    def _prefix(self, identifier, values):
        return {
            "id": identifier,
            "actions": {
                action: {
                    "em": float(value),
                    "f1": float(value),
                    "source_recall": float(value),
                    "reader_latency_seconds": 1.0,
                    "reader_generated_tokens": 10,
                }
                for action, value in zip(ACTIONS, values, strict=True)
            },
        }

    def test_final_and_router_use_different_answers_for_reason(self):
        prefixes = [self._prefix("a", (0, 0, 0.4, 0.9)), self._prefix("b", (0, 0, 0.5, 0.2))]
        predictions = [
            {"id": "a", "route": "SEARCH", "searches": 3, "em": 0.9, "f1": 0.9},
            {"id": "b", "route": "REASON", "searches": 2, "em": 1.0, "f1": 1.0},
        ]
        vectors = _vectors(prefixes, predictions)
        self.assertAlmostEqual(vectors["router_f1"].mean(), 0.7)
        self.assertAlmostEqual(vectors["final_f1"].mean(), 0.95)
        self.assertEqual(vectors["router_search"].tolist(), [1.0, 0.0])

    def test_matched_random_has_exact_search_budget_and_expected_value(self):
        prefixes = [self._prefix(str(i), (0, 0, 0.0, float(i % 2))) for i in range(4)]
        predictions = [
            {"id": str(i), "route": "SEARCH" if i < 2 else "STOP", "searches": 3 if i < 2 else 2, "em": 0.0, "f1": 0.0}
            for i in range(4)
        ]
        vectors = _vectors(prefixes, predictions)
        report = _random_matched_budget(vectors, 100, 100, 7)
        self.assertEqual(report["search_count"], 2)
        self.assertEqual(report["average_searches"], 2.5)
        self.assertAlmostEqual(report["f1"]["random_expected"], 0.25)

    def test_frozen_majority_routes_without_gold(self):
        bundles = [
            {
                "evidence_model": _Probabilities([0.9, 0.1, 0.1]),
                "answer_model": _Probabilities([0.1, 0.9, 0.1]),
                "search_threshold": 0.5,
                "answer_threshold": 0.5,
            }
            for _ in range(3)
        ]
        routes = _routes(np.zeros((3, 1)), np.zeros((3, 1)), bundles)
        self.assertEqual([route for route, _ in routes], ["SEARCH", "REASON", "STOP"])

    def test_three_way_random_matches_search_and_reason_budgets(self):
        prefixes = [self._prefix(str(i), (0, 0, 0.0, float(i % 2))) for i in range(3)]
        predictions = [
            {"id": "0", "route": "SEARCH", "searches": 3, "em": 0.0, "f1": 0.0},
            {"id": "1", "route": "REASON", "searches": 2, "em": 1.0, "f1": 1.0},
            {"id": "2", "route": "STOP", "searches": 2, "em": 0.0, "f1": 0.0},
        ]
        judged = {
            str(i): {"judged_em": float(i == 1), "judged_f1": float(i == 1)}
            for i in range(3)
        }
        vectors = _vectors(prefixes, predictions, judged)
        report = _random_three_way(vectors, 100, 100, 7)
        self.assertEqual((report["search_count"], report["reason_count"], report["stop_count"]), (1, 1, 1))
        self.assertAlmostEqual(report["average_searches"], 7 / 3)
        self.assertAlmostEqual(report["average_reader_calls"], 5 / 3)

    def test_frozen_router_predictions_are_one_complete_id_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.jsonl"
            path.write_text(
                "\n".join(json.dumps({"id": identifier, "route": "STOP"}) for identifier in ("a", "b")) + "\n",
                encoding="utf-8",
            )
            maps = _seed_predictions(Path(directory), (42, 43, 44), str(path))
            self.assertEqual(len(maps), 1)
            self.assertEqual(set(maps[0]), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
