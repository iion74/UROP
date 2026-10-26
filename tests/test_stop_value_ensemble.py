import unittest
from types import SimpleNamespace

import numpy as np

from experiments.evaluate_stop_value_ensemble import _calibrate, _simulate, _vote_matrix


class StopValueEnsembleTests(unittest.TestCase):
    def test_vote_matrix_uses_oof_row_thresholds(self):
        rows = [{"id": "a"}]
        predictions = (
            {"a": {"predicted_advantages": [1, 2, 3], "threshold": [0, 3, 3]}},
            {"a": {"predicted_advantages": [-1, 4, 2], "threshold": [0, 3, 3]}},
        )
        self.assertEqual(_vote_matrix(rows, predictions).tolist(), [[1, 1, 0]])

    def test_simulation_stops_at_first_failed_vote_threshold(self):
        scores = np.asarray([[5, 2, 5], [5, 5, 1], [5, 5, 5]])
        self.assertEqual(_simulate(scores, (3, 3, 3)).tolist(), [1, 2, 3])

    def test_train_calibration_does_not_require_validation_rows(self):
        actions = {
            "vector": {"em": 0.0, "f1": 0.0},
            "planned_continue_1": {"em": 1.0, "f1": 1.0},
            "planned_continue_2": {"em": 1.0, "f1": 1.0},
            "planned_continue_3": {"em": 1.0, "f1": 1.0},
        }
        rows = [
            {"id": "a", "dataset": "hotpotqa", "direct_em": 0.0, "direct_f1": 0.0, "actions": actions}
        ]
        votes = np.asarray([[5, 0, 0]])
        rule = np.zeros((1, 3), dtype="int64")
        args = SimpleNamespace(
            mode="ensemble",
            max_rule_weight=5,
            datasets=("hotpotqa",),
            em_weight=0.5,
            f1_weight=0.5,
            search_cost=0.0,
            dataset_harm_penalty=2.0,
        )
        _, thresholds, audit = _calibrate(rows, votes, rule, args)
        self.assertEqual(_simulate(votes, thresholds).tolist(), [1])
        self.assertEqual(audit["metrics"]["hotpotqa"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
