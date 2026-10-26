import unittest

from src.causal_evidence import (
    FEATURE_NAMES,
    POLICY_FEATURE_NAMES,
    WIDTH_FEATURE_NAMES,
    evidence_features,
    evidence_width_features,
    proof_policy_features,
    select_topology_aware_indices,
    select_evidence_indices,
)


class Document:
    def __init__(self, title, text):
        self.title = title
        self.text = text


class CausalEvidenceTests(unittest.TestCase):
    def test_features_capture_cross_document_title_bridge(self) -> None:
        candidates = [
            {"content_id": "a", "title": "Alpha", "score": 0.8},
            {"content_id": "b", "title": "Beta", "score": 0.7},
        ]
        documents = {
            "a": Document("Alpha", "Alpha was created by Beta."),
            "b": Document("Beta", "Beta was born in Seoul."),
        }
        matrix = evidence_features("Where was Alpha's creator born?", candidates, documents)

        self.assertEqual(matrix.shape, (2, len(FEATURE_NAMES)))
        outgoing = FEATURE_NAMES.index("outgoing_title_mentions")
        incoming = FEATURE_NAMES.index("incoming_title_mentions")
        self.assertEqual(matrix[0, outgoing], 1.0)
        self.assertEqual(matrix[1, incoming], 1.0)

    def test_selection_is_variable_sized_and_preserves_source_order(self) -> None:
        selected = select_evidence_indices([0.1, 0.8, 0.7, 0.2, 0.6], maximum=4, threshold=0.5)
        self.assertEqual(selected, (1, 2, 4))
        self.assertEqual(select_evidence_indices([0.2, 0.1], minimum=2), (0, 1))

    def test_proof_policy_adds_one_answer_action_without_gold_features(self) -> None:
        candidates = [
            {"content_id": "a", "title": "Alpha", "score": 0.8},
            {"content_id": "b", "title": "Beta", "score": 0.7},
        ]
        documents = {
            "a": Document("Alpha", "Alpha was created by Beta."),
            "b": Document("Beta", "Beta was born in Seoul."),
        }
        matrix = proof_policy_features("Where was Alpha's creator born?", candidates, documents, [0])

        self.assertEqual(matrix.shape, (3, len(POLICY_FEATURE_NAMES)))
        answer_column = POLICY_FEATURE_NAMES.index("action_is_answer")
        self.assertEqual(matrix[:, answer_column].tolist(), [0.0, 0.0, 1.0])
        self.assertFalse(any("dataset" in name or "gold" in name for name in POLICY_FEATURE_NAMES))

    def test_width_features_are_dataset_blind(self) -> None:
        class FakeModel:
            def predict_proba(self, matrix):
                import numpy as np

                positive = np.linspace(0.8, 0.2, len(matrix))
                return np.column_stack((1.0 - positive, positive))

        candidates = [
            {"content_id": "a", "title": "Alpha", "score": 0.8},
            {"content_id": "b", "title": "Beta", "score": 0.7},
        ]
        documents = {
            "a": Document("Alpha", "Alpha was created by Beta."),
            "b": Document("Beta", "Beta was born in Seoul."),
        }
        features = evidence_width_features(
            FakeModel(), "Where was Alpha's creator born?", candidates, documents
        )
        self.assertEqual(features.shape, (len(WIDTH_FEATURE_NAMES),))
        self.assertFalse(any("dataset" in name or "gold" in name for name in WIDTH_FEATURE_NAMES))

    def test_topology_selection_preserves_two_parallel_branches(self) -> None:
        candidates = [
            {"content_id": "fa", "title": "Film Alpha"},
            {"content_id": "pa", "title": "Director Alpha"},
            {"content_id": "fb", "title": "Film Beta"},
            {"content_id": "pb", "title": "Director Beta"},
            {"content_id": "noise", "title": "Noise"},
        ]
        documents = {
            "fa": Document("Film Alpha", "Film Alpha was directed by Director Alpha."),
            "pa": Document("Director Alpha", "Director Alpha was born in 1950."),
            "fb": Document("Film Beta", "Film Beta was directed by Director Beta."),
            "pb": Document("Director Beta", "Director Beta was born in 1960."),
            "noise": Document("Noise", "Unrelated."),
        }
        selected = select_topology_aware_indices(
            "Which film has the older director, Film Alpha or Film Beta?",
            candidates,
            documents,
            [0.9, 0.8, 0.4, 0.3, 0.7],
            width=3,
        )
        self.assertEqual(set(selected), {0, 1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()
