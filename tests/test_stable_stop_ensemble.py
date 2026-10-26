import unittest

import numpy as np

from experiments.train_stable_stop_ensemble import _prediction_matrix, _simulate


class StableStopEnsembleTests(unittest.TestCase):
    def test_prediction_matrix_restores_question_step_order(self):
        rows = [{"id": "a"}, {"id": "b"}]
        keys = [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2)]
        flat = np.asarray([1, 2, 3, 4, 5, 6], dtype="float64")
        self.assertEqual(_prediction_matrix(rows, flat, keys).tolist(), [[1, 2, 3], [4, 5, 6]])

    def test_simulate_uses_first_gain_below_threshold(self):
        predictions = np.asarray([[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertEqual(_simulate(predictions, (0.0, 0.0, 0.0)).tolist(), [1, 3])


if __name__ == "__main__":
    unittest.main()
