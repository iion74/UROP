import unittest

import numpy as np

from experiments.evaluate_frozen_action_gain_full import (
    _policy_report,
    _route_from_gains,
    _scores,
)


class FrozenActionGainFullTests(unittest.TestCase):
    def test_route_uses_only_predicted_signed_f1_gain(self):
        routes = _route_from_gains(
            np.asarray([-0.2, 0.1, 0.2, 0.2, 0.0]),
            np.asarray([-0.1, 0.3, 0.2, -0.4, 0.0]),
        )
        self.assertEqual(routes.tolist(), ["STOP", "REASON", "SEARCH", "SEARCH", "STOP"])

    def test_policy_accounts_for_answer_before_extra_search_and_two_reason_calls(self):
        data = {
            "ids": ["a", "b", "c"],
            "f1_2": np.asarray([0.0, 0.0, 1.0]),
            "f1_3": np.asarray([1.0, 0.0, 0.0]),
            "em_2": np.asarray([0.0, 0.0, 1.0]),
            "em_3": np.asarray([1.0, 0.0, 0.0]),
            "latency_2": np.asarray([1.0, 1.0, 1.0]),
            "latency_3": np.asarray([2.0, 2.0, 2.0]),
            "reader_2_latency": np.asarray([0.4, 0.4, 0.4]),
            "tokens_2": np.asarray([10.0, 10.0, 10.0]),
            "tokens_3": np.asarray([20.0, 20.0, 20.0]),
        }
        reason = {
            "f1": np.asarray([0.0, 1.0, 1.0]),
            "em": np.asarray([0.0, 1.0, 1.0]),
            "extra_latency": np.asarray([0.7, 0.7, 0.7]),
            "extra_tokens": np.asarray([8.0, 8.0, 8.0]),
            "extra_reader_calls": np.asarray([2.0, 2.0, 2.0]),
        }
        routes = np.asarray(["SEARCH", "REASON", "STOP"])
        scores = _scores(routes, data, reason)
        self.assertEqual(scores["f1"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(scores["latency"].tolist(), [2.4, 1.7, 1.0])
        self.assertEqual(scores["tokens"].tolist(), [30.0, 18.0, 10.0])
        old = {"f1": data["f1_3"], "em": data["em_3"]}
        report = _policy_report(routes, data, reason, old, bootstrap=100, draws=100, seed=7)
        self.assertAlmostEqual(report["f1"], 100.0)
        self.assertAlmostEqual(report["average_additional_searches"], 2 + 1 / 3)
        self.assertAlmostEqual(report["average_reader_calls"], 2.0)
        self.assertAlmostEqual(report["matched_random_expected_f1"], 100 * 4 / 9)


if __name__ == "__main__":
    unittest.main()
