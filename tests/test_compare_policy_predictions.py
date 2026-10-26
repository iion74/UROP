import unittest

from experiments.compare_policy_predictions import _compare


class ComparePolicyPredictionsTests(unittest.TestCase):
    def test_candidate_minus_reference_is_paired_by_id(self):
        reference = {
            "a": {"id": "a", "dataset": "hotpotqa", "policy_em": 0.0, "policy_f1": 0.2, "stop_step": 1},
            "b": {"id": "b", "dataset": "hotpotqa", "policy_em": 1.0, "policy_f1": 0.8, "stop_step": 2},
        }
        candidate = {
            "b": {"id": "b", "dataset": "hotpotqa", "policy_em": 1.0, "policy_f1": 1.0, "stop_step": 2},
            "a": {"id": "a", "dataset": "hotpotqa", "policy_em": 1.0, "policy_f1": 0.4, "stop_step": 2},
        }

        result = _compare(reference, candidate, ("hotpotqa",), 100, 42, 0.002)["hotpotqa"]

        self.assertAlmostEqual(result["candidate_minus_reference"]["em"]["observed_difference"], 0.5)
        self.assertAlmostEqual(result["candidate_minus_reference"]["f1"]["observed_difference"], 0.2)
        self.assertAlmostEqual(result["candidate_minus_reference"]["average_extra_searches"], 0.5)


if __name__ == "__main__":
    unittest.main()
