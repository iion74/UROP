import unittest

import numpy as np

from experiments.evaluate_answer_gain_search_policy import _route_summary, _threshold


class AnswerGainSearchPolicyTests(unittest.TestCase):
    def test_threshold_uses_signed_answer_gain(self):
        gain = np.asarray([1.0, 0.5, -0.5, -1.0])
        prediction = gain.copy()
        threshold, report = _threshold(prediction, gain, 0.0)
        chosen = prediction > threshold
        self.assertEqual(chosen.tolist(), [True, True, False, False])
        self.assertAlmostEqual(report["oof_policy_gain_points"], 37.5)

    def test_reader_aware_search_counts_extra_reader_call_and_tokens(self):
        data = {
            "f1_2": np.asarray([0.0, 0.0, 1.0, 1.0]),
            "f1_3": np.asarray([1.0, 0.0, 0.0, 1.0]),
            "em_2": np.asarray([0.0, 0.0, 1.0, 1.0]),
            "em_3": np.asarray([1.0, 0.0, 0.0, 1.0]),
            "latency_2": np.ones(4),
            "latency_3": np.ones(4) * 2,
            "reader_2_latency": np.ones(4) * 0.5,
            "tokens_2": np.ones(4) * 10,
            "tokens_3": np.ones(4) * 20,
        }
        mask = np.asarray([True, False, False, False])
        free = _route_summary(data, mask, bootstrap=100, draws=100, seed=42, reader_aware=False)
        aware = _route_summary(data, mask, bootstrap=100, draws=100, seed=42, reader_aware=True)
        self.assertEqual(free["search_count"], 1)
        self.assertAlmostEqual(free["f1"], 75.0)
        self.assertAlmostEqual(free["matched_random_expected_f1"], 50.0)
        self.assertAlmostEqual(free["average_reader_calls"], 1.0)
        self.assertAlmostEqual(aware["average_reader_calls"], 1.25)
        self.assertAlmostEqual(aware["reader_generated_tokens"], 15.0)
        self.assertAlmostEqual(aware["estimated_latency_seconds"], 1.375)


if __name__ == "__main__":
    unittest.main()
