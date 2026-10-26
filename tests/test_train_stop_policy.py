import unittest

from experiments.train_stop_policy import _numeric_state, _simulate


class TrainStopPolicyTests(unittest.TestCase):
    def test_numeric_state_contains_no_gold_hop(self):
        values = _numeric_state(
            {
                "completed_steps": 1,
                "reader_source_count": 5,
                "memory_size": 20,
                "previous_query_count": 1,
                "last_added_score": 0.8,
                "consecutive_no_new": 0,
                "gold_hop": 4,
            },
            ["same query", "same query"],
        )
        self.assertEqual(len(values), 9)
        self.assertEqual(values[-1], 1.0)

    def test_simulation_stops_at_first_probability_below_threshold(self):
        row = {
            "id": "q1",
            "dataset": "musique",
            "direct_em": 0.0,
            "direct_f1": 0.0,
            "oracle_em": 1.0,
            "oracle_f1": 1.0,
            "actions": {
                "vector": {"em": 0.0, "f1": 0.0},
                "planned_continue_1": {"em": 0.0, "f1": 0.5},
                "planned_continue_2": {"em": 1.0, "f1": 1.0},
                "planned_continue_3": {"em": 0.0, "f1": 0.5},
            },
        }
        record = _simulate([row], {"q1": [0.9, 0.4, 0.9]}, 0.5)[0]
        self.assertEqual(record["stop_step"], 1)
        self.assertEqual(record["selected_action"], "planned_continue_1")


if __name__ == "__main__":
    unittest.main()
