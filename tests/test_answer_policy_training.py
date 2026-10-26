import unittest

import numpy as np

from experiments.train_answer_policy import _one_shot_values, _sequential_values


class AnswerPolicyTrainingTests(unittest.TestCase):
    def test_one_shot_threshold_falls_back_to_direct(self) -> None:
        row = {
            "direct_f1": 0.5,
            "actions": {
                "vector": {"f1": 0.5},
                "graph_1hop_small": {"f1": 0.6},
                "graph_1hop_large": {"f1": 0.4},
                "graph_2hop_small": {"f1": 0.7},
                "graph_2hop_large": {"f1": 0.3},
            },
        }

        values, actions = _one_shot_values(
            [row], np.asarray([[0.1, -0.1, 0.2, -0.2]]), 0.25, return_actions=True
        )

        self.assertEqual(actions, ["vector"])
        self.assertEqual(values, [0.5])

    def test_sequential_uses_state_prediction_after_selected_first_step(self) -> None:
        row = {
            "sequential": {
                "states_after_first_step": {
                    "graph_1hop_small": {
                        "action_trajectories": {
                            "stop": "small_stop",
                            "return_direct": "small_return_direct",
                            "expand_2hop_small": "small_expand_2hop_small",
                            "expand_2hop_large": "small_expand_2hop_large",
                        }
                    },
                    "graph_1hop_large": {
                        "action_trajectories": {
                            "stop": "large_stop",
                            "return_direct": "large_return_direct",
                            "expand_2hop_small": "large_expand_2hop_small",
                            "expand_2hop_large": "large_expand_2hop_large",
                        }
                    },
                },
                "terminal_results": {
                    "stop": {"f1": 0.5},
                    "small_stop": {"f1": 0.4},
                    "small_return_direct": {"f1": 0.5},
                    "small_expand_2hop_small": {"f1": 0.8},
                    "small_expand_2hop_large": {"f1": 0.6},
                    "large_stop": {"f1": 0.3},
                    "large_return_direct": {"f1": 0.5},
                    "large_expand_2hop_small": {"f1": 0.2},
                    "large_expand_2hop_large": {"f1": 0.1},
                },
            }
        }

        values, paths = _sequential_values(
            [row],
            np.asarray([[0.5, 0.1]]),
            np.asarray([[0.6, 0.2], [0.1, 0.1]]),
            0.3,
            0.3,
        )

        self.assertEqual(paths, ["small_expand_2hop_small"])
        self.assertEqual(values, [0.8])

        values, paths = _sequential_values(
            [row],
            np.asarray([[0.5, 0.1]]),
            np.asarray([[0.7, 0.1, 0.0], [0.1, 0.1, 0.1]]),
            0.3,
            0.3,
            post_actions=("return_direct", "expand_2hop_small", "expand_2hop_large"),
        )

        self.assertEqual(paths, ["small_return_direct"])
        self.assertEqual(values, [0.5])


if __name__ == "__main__":
    unittest.main()
