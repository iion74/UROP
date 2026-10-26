import unittest

import numpy as np

from experiments.evaluate_reason_gain_gate import _summary


class ReasonGainGateTests(unittest.TestCase):
    def test_skipped_reason_reverts_to_baseline_and_saves_cost(self):
        data = {
            "ids": ["a", "b", "c", "d"],
            "gain": np.asarray([1.0, -1.0, 0.0, 0.0]),
            "baseline_f1": np.asarray([0.0, 1.0, 0.0, 1.0]),
            "baseline_em": np.asarray([0.0, 1.0, 0.0, 1.0]),
            "judged_f1": np.asarray([1.0, 0.0, 0.0, 1.0]),
            "judged_em": np.asarray([1.0, 0.0, 0.0, 1.0]),
            "reason_latency": np.ones(4) * 2.0,
            "reason_tokens": np.ones(4) * 20.0,
        }
        old = {identifier: {"f1": float(score), "em": float(score)}
               for identifier, score in zip(data["ids"], data["judged_f1"], strict=True)}
        keep = np.asarray([True, False, False, False])
        result = _summary(np.asarray([1.0, -1.0, 0.0, 0.0]), data, old, keep, random_draws=100)
        self.assertEqual((result["reason_kept"], result["reason_skipped"]), (1, 3))
        self.assertAlmostEqual(result["queue_f1"], 75.0)
        self.assertAlmostEqual(result["queue_f1_change_vs_old_points"], 25.0)
        self.assertAlmostEqual(result["saved_latency_seconds_per_queue_question"], 1.5)
        self.assertAlmostEqual(result["saved_reader_generated_tokens_per_queue_question"], 15.0)


if __name__ == "__main__":
    unittest.main()
