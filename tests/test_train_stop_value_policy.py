import unittest
from types import SimpleNamespace

from experiments.train_stop_value_policy import (
    GraphStateIndex,
    _hop_balanced_objective,
    _simulate,
    _state_text,
    _value_numeric,
)


class TrainStopValuePolicyTests(unittest.TestCase):
    def test_value_numeric_contains_retrieval_dynamics_without_gold(self):
        values = _value_numeric(
            {
                "completed_steps": 1,
                "reader_source_count": 5,
                "memory_size": 20,
                "previous_query_count": 1,
                "last_added_score": 0.8,
                "consecutive_no_new": 0,
                "gold_hop": 4,
            },
            ["bridge query"],
            [
                {"content_id": "a", "score": 0.9},
                {"content_id": "b", "score": 0.6},
            ],
            [{"content_id": "a", "score": 0.9}],
        )
        self.assertEqual(len(values), 23)
        self.assertAlmostEqual(values[9], 0.9)
        self.assertAlmostEqual(values[13], 0.3)
        self.assertAlmostEqual(values[14], 0.5)
        self.assertAlmostEqual(values[15], 0.5)

    def test_value_numeric_includes_graph_frontier_and_internal_connectivity(self):
        graph = GraphStateIndex(
            graph={0: (1, 2), 1: (0, 2), 2: (0, 1)},
            positions_by_source={"a": (0,), "b": (1,), "c": (2,)},
            source_by_position=("a", "b", "c"),
        )
        values = _value_numeric(
            {"reader_source_count": 2, "memory_size": 2},
            [],
            [{"content_id": "a", "score": 0.9}, {"content_id": "b", "score": 0.8}],
            [{"content_id": "a", "score": 0.9}],
            graph,
        )
        self.assertGreater(values[17], 0.0)
        self.assertGreater(values[18], 0.0)
        self.assertGreater(values[19], 0.0)

    def test_state_text_contains_current_evidence_but_not_answer(self):
        text = _state_text(
            "Who wrote the book?",
            ["book author"],
            [{"content_id": "doc-1", "title": "Fallback"}],
            {"doc-1": {"title": "Book", "text": "The book was published in 1990."}},
            200,
        )
        self.assertIn("Who wrote the book?", text)
        self.assertIn("book author", text)
        self.assertIn("Book: The book was published", text)

    def test_simulation_stops_when_continue_advantage_is_not_large_enough(self):
        row = {
            "id": "q1",
            "dataset": "musique",
            "direct_em": 0.0,
            "direct_f1": 0.0,
            "oracle_f1": 1.0,
            "actions": {
                "vector": {"em": 0.0, "f1": 0.0},
                "planned_continue_1": {"em": 0.0, "f1": 0.5},
                "planned_continue_2": {"em": 1.0, "f1": 1.0},
                "planned_continue_3": {"em": 0.0, "f1": 0.5},
            },
        }
        record = _simulate([row], {"q1": [0.3, 0.04, 0.8]}, threshold=0.05)[0]
        self.assertEqual(record["stop_step"], 1)
        self.assertEqual(record["selected_action"], "planned_continue_1")

        step_specific = _simulate(
            [row],
            {"q1": [0.3, 0.04, 0.8]},
            threshold=(0.2, 0.0, 0.9),
        )[0]
        self.assertEqual(step_specific["stop_step"], 2)
        self.assertEqual(step_specific["selected_action"], "planned_continue_2")

    def test_hop_balanced_objective_does_not_let_large_group_hide_harm(self):
        rows = [
            {"id": "a", "dataset": "musique", "hop_count": 2, "direct_em": 0.5, "direct_f1": 0.5},
            {"id": "b", "dataset": "musique", "hop_count": 2, "direct_em": 0.5, "direct_f1": 0.5},
            {"id": "c", "dataset": "musique", "hop_count": 4, "direct_em": 0.5, "direct_f1": 0.5},
        ]
        args = SimpleNamespace(f1_weight=0.5, em_weight=0.5, dataset_harm_penalty=2.0)

        def records(values):
            return [
                {"id": row["id"], "policy_em": value, "policy_f1": value}
                for row, value in zip(rows, values, strict=True)
            ]

        safe = _hop_balanced_objective(rows, records([0.5, 0.5, 0.5]), args)
        harmful = _hop_balanced_objective(rows, records([0.6, 0.6, 0.0]), args)
        self.assertGreater(safe, harmful)


if __name__ == "__main__":
    unittest.main()
