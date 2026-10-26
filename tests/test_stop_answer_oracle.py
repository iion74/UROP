import unittest

from experiments.evaluate_stop_answer_oracle import (
    _answer_stop_labels,
    _order_sources_for_reader,
    _restore_sources,
    _trajectory_actions,
)
from src.document_loader import Chunk


class StopAnswerOracleTests(unittest.TestCase):
    def test_initial_first_reader_order_preserves_initial_rank_then_additions(self) -> None:
        initial_a = (Chunk("a", "da", "A", "A"), 0.9)
        initial_b = (Chunk("b", "db", "B", "B"), 0.8)
        addition_c = (Chunk("c", "dc", "C", "C"), 0.7)
        addition_d = (Chunk("d", "dd", "D", "D"), 0.6)

        ordered = _order_sources_for_reader(
            [initial_a, addition_c, initial_b, addition_d],
            ("a", "b"),
            "initial_first",
        )

        self.assertEqual([chunk.chunk_id for chunk, _ in ordered], ["a", "b", "c", "d"])

    def test_four_hop_actions_are_required_in_order(self):
        row = {
            "actions": {
                "vector": {},
                "planned_continue_1": {},
                "planned_continue_2": {},
                "planned_continue_3": {},
            }
        }
        self.assertEqual(
            _trajectory_actions(row),
            ("vector", "planned_continue_1", "planned_continue_2", "planned_continue_3"),
        )

    def test_direct_sources_can_be_restored_without_scoring_direct_action(self):
        initial = Chunk("initial", "doc", "Initial", "Initial evidence")
        restored = _restore_sources(
            {"evidence": [{"chunk_id": "initial", "score": 0.75}]},
            {"initial": initial},
            {},
        )

        self.assertEqual([(chunk.chunk_id, score) for chunk, score in restored], [("initial", 0.75)])

    def test_answer_f1_future_gain_creates_continue_then_stop(self):
        actions = ("vector", "planned_continue_1", "planned_continue_2", "planned_continue_3")
        results = {
            "vector": {"f1": 0.0, "em": 0.0},
            "planned_continue_1": {"f1": 0.5, "em": 0.0},
            "planned_continue_2": {"f1": 1.0, "em": 1.0},
            "planned_continue_3": {"f1": 0.5, "em": 0.0},
        }
        labels = _answer_stop_labels(results, actions)
        self.assertEqual([row["label"] for row in labels], ["CONTINUE", "CONTINUE", "STOP"])


if __name__ == "__main__":
    unittest.main()
