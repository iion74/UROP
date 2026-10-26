import unittest

import numpy as np

from src.answer_policy import QuestionEvidenceGate


class QuestionEvidenceGateTests(unittest.TestCase):
    def test_gate_combines_question_text_and_numeric_evidence(self) -> None:
        numeric = np.asarray(
            [
                [0.1, 0.9],
                [0.2, 0.8],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.15, 0.85],
                [0.85, 0.15],
            ]
        )
        questions = [
            "which bridge entity",
            "which bridge person",
            "simple direct entity",
            "simple direct person",
            "which bridge answer",
            "simple direct answer",
        ]
        labels = np.asarray([1, 1, 0, 0, 1, 0])

        gate = QuestionEvidenceGate.fit(numeric, questions, labels, seed=7)
        probabilities = gate.predict_graph_probability(numeric, questions)

        self.assertEqual(probabilities.shape, (6,))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertGreater(probabilities[labels == 1].mean(), probabilities[labels == 0].mean())

    def test_gate_rejects_wrong_numeric_dimension(self) -> None:
        numeric = np.asarray([[0.1], [0.9], [0.2], [0.8]])
        questions = ["graph bridge", "direct answer", "graph entity", "direct entity"]
        labels = np.asarray([1, 0, 1, 0])
        gate = QuestionEvidenceGate.fit(numeric, questions, labels)

        with self.assertRaises(ValueError):
            gate.predict_graph_probability(np.asarray([[0.1, 0.2]]), ["question"])


if __name__ == "__main__":
    unittest.main()
