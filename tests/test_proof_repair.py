from types import SimpleNamespace
import unittest

import numpy as np

from src.proof_repair import (
    REPAIR_FEATURE_NAMES,
    SYNDROME_FEATURE_NAMES,
    build_graph_repair_proposals,
    graph_repair_features,
    repair_syndrome_features,
    select_graph_repairs,
)


class FakeRetriever:
    def __init__(self):
        self.chunk_positions = {"a:0": 0, "b:0": 1, "c:0": 2}
        self.chunks = [
            SimpleNamespace(chunk_id="a:0", source="a", title="Alpha"),
            SimpleNamespace(chunk_id="b:0", source="b", title="Beta"),
            SimpleNamespace(chunk_id="c:0", source="c", title="Gamma"),
        ]
        self.graph = {0: [2], 1: [2], 2: [0, 1]}

    def _similarity(self, first, second):
        return 0.8 if {first, second} == {0, 2} else 0.6


def _documents():
    return {
        "a": SimpleNamespace(title="Alpha", text="Alpha points to Gamma."),
        "b": SimpleNamespace(title="Beta", text="Beta is unrelated."),
        "c": SimpleNamespace(title="Gamma", text="Gamma contains the requested value."),
    }


class ProofRepairTests(unittest.TestCase):
    def test_graph_repair_proposals_keep_unselected_and_merge_neighbor_edges(self):
        candidates = [
            {"chunk_id": "a:0", "content_id": "a", "title": "Alpha", "score": 0.9},
            {"chunk_id": "b:0", "content_id": "b", "title": "Beta", "score": 0.7},
        ]
        proposals = build_graph_repair_proposals(FakeRetriever(), candidates, [0])
        self.assertEqual([proposal["content_id"] for proposal in proposals], ["b", "c"])
        self.assertEqual(proposals[1]["seed_indices"], [0, 1])

    def test_graph_repair_features_and_bounded_selection(self):
        candidates = [
            {"chunk_id": "a:0", "content_id": "a", "title": "Alpha", "score": 0.9},
            {"chunk_id": "b:0", "content_id": "b", "title": "Beta", "score": 0.7},
        ]
        proposals = build_graph_repair_proposals(FakeRetriever(), candidates, [0])
        features = graph_repair_features(
            "What value is linked from Alpha?",
            candidates,
            [0],
            [0.9, 0.2],
            proposals,
            _documents(),
        )
        self.assertEqual(features.shape, (2, len(REPAIR_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(
            select_graph_repairs([0.4, 0.9], proposals, threshold=0.5, maximum_additions=1),
            (1,),
        )
        syndrome = repair_syndrome_features(features, [0.4, 0.9], selected_document_count=1)
        self.assertEqual(syndrome.shape, (len(SYNDROME_FEATURE_NAMES),))
        self.assertTrue(np.isfinite(syndrome).all())


if __name__ == "__main__":
    unittest.main()
