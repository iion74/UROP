import unittest

import numpy as np

from experiments.evaluate_rule_stop_policy import RuleBatch, _rule_observation, _simulate_steps


def _batch(scores, hard_stop=None):
    count = len(scores)
    hard_stop = hard_stop or [[False, False, False] for _ in range(count)]
    zeros = np.zeros((count, 4), dtype="float64")
    return RuleBatch(
        rows=tuple({"id": str(index)} for index in range(count)),
        scores=np.asarray(scores, dtype="float64"),
        hard_stop=np.asarray(hard_stop, dtype="bool"),
        action_em=zeros,
        action_f1=zeros,
        action_source_recall=zeros,
        action_all_support_recall=zeros,
        datasets=np.asarray(["hotpotqa"] * count, dtype="object"),
    )


class RuleStopPolicyTests(unittest.TestCase):
    def test_initial_confidence_can_stop_without_expansion(self):
        batch = _batch([[0.8, 0.7, 0.7], [0.4, 0.7, 0.7]])

        steps = _simulate_steps(batch, (0.6, 0.5, 0.5))

        self.assertEqual(steps.tolist(), [0, 3])

    def test_weak_added_score_stops_at_first_expansion(self):
        batch = _batch([[0.4, 0.3, 0.9]])

        steps = _simulate_steps(batch, (0.6, 0.5, 0.5))

        self.assertEqual(steps.tolist(), [1])

    def test_hard_stop_overrides_score_rule(self):
        batch = _batch([[0.4, 0.9, 0.9]], [[False, True, False]])

        steps = _simulate_steps(batch, (0.6, 0.5, 0.5))

        self.assertEqual(steps.tolist(), [1])

    def test_observation_uses_second_initial_and_previous_added_scores(self):
        row = {
            "planner_audit": [
                {"query": "first", "evidence_sufficient": None, "policy_state": {}},
                {
                    "query": "first",
                    "evidence_sufficient": None,
                    "policy_state": {"last_added_score": 0.61, "consecutive_no_new": 0},
                },
                {
                    "query": "second",
                    "evidence_sufficient": None,
                    "policy_state": {"last_added_score": 0.52, "consecutive_no_new": 0},
                },
            ]
        }
        trajectory = {
            "actions": {
                "vector": {"evidence": [{"score": 0.9}, {"score": 0.7}, {"score": 0.8}]}
            }
        }

        scores, hard_stop = _rule_observation(row, trajectory)

        self.assertEqual(scores, (0.8, 0.61, 0.52))
        self.assertEqual(hard_stop, (False, False, True))


if __name__ == "__main__":
    unittest.main()
