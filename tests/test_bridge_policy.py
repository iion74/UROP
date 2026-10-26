import unittest

import numpy as np

from experiments.train_bridge_policy import (
    DIRECT_ACTION,
    NON_DIRECT_ACTIONS,
    Thresholds,
    _select,
)


class BridgePolicyTests(unittest.TestCase):
    def test_selection_requires_gate_and_positive_predicted_advantage(self) -> None:
        rows = []
        for index in range(3):
            actions = {DIRECT_ACTION: {"em": 0.0, "f1": 0.2}}
            actions.update({action: {"em": 1.0, "f1": 0.8} for action in NON_DIRECT_ACTIONS})
            rows.append({"actions": actions})
        probabilities = np.asarray([0.9, 0.4, 0.9])
        advantages = np.zeros((3, len(NON_DIRECT_ACTIONS)), dtype="float64")
        advantages[:, 0] = [0.2, 0.2, 0.01]

        actions, _, f1_values = _select(
            rows,
            probabilities,
            advantages,
            Thresholds(gate=0.5, advantage=0.05),
        )

        self.assertEqual(actions, [NON_DIRECT_ACTIONS[0], DIRECT_ACTION, DIRECT_ACTION])
        self.assertEqual(f1_values, [0.8, 0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
