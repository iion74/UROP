import unittest

import faiss
import numpy as np
from pathlib import Path
import tempfile

from src.document_loader import Chunk
from src.retriever import SemanticRetriever, _matched_title_aliases
from src.routing import EvidenceProfile, QueryRouter


class EvidenceGainRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = QueryRouter(small_budget=3, large_budget=6, utility_threshold=0.12)

    def test_concentrated_direct_evidence_stops_before_graph_expansion(self) -> None:
        profile = EvidenceProfile(
            score_concentration=0.90,
            score_margin=0.85,
            source_dispersion=0.10,
            graph_connectivity=0.10,
            redundancy=0.80,
            one_hop_gain=0.15,
            two_hop_gain=0.10,
        )

        decision = self.router.decide(profile)

        self.assertEqual(decision.route, "vector")
        self.assertEqual(decision.graph_hops, 0)
        self.assertEqual(decision.expansion_budget, 0)
        self.assertIn("gain_below_graph_cost", decision.reasons)

    def test_uncertain_distributed_evidence_uses_one_hop_graph(self) -> None:
        profile = EvidenceProfile(
            score_concentration=0.10,
            score_margin=0.10,
            source_dispersion=0.80,
            graph_connectivity=0.60,
            redundancy=0.40,
            one_hop_gain=0.80,
            two_hop_gain=0.20,
        )

        decision = self.router.decide(profile)

        self.assertEqual(decision.route, "graph_1hop")
        self.assertEqual(decision.graph_hops, 1)
        self.assertEqual(decision.expansion_budget, 6)
        self.assertIn("1hop_gain_exceeds_cost", decision.reasons)

    def test_second_hop_requires_its_own_marginal_gain(self) -> None:
        profile = EvidenceProfile(
            score_concentration=0.10,
            score_margin=0.10,
            source_dispersion=0.80,
            graph_connectivity=0.70,
            redundancy=0.40,
            one_hop_gain=0.20,
            two_hop_gain=0.95,
        )

        decision = self.router.decide(profile)

        self.assertEqual(decision.route, "graph_2hop")
        self.assertEqual(decision.graph_hops, 2)
        self.assertIn("second_hop_has_marginal_gain", decision.reasons)

    def test_graph_gain_ignores_candidates_already_in_direct_results(self) -> None:
        retriever = SemanticRetriever(index_dir=None)  # type: ignore[arg-type]
        retriever.chunks = [
            Chunk("0", "a", "first", "first"),
            Chunk("1", "b", "second", "second"),
            Chunk("2", "c", "third", "third"),
        ]
        retriever.chunk_positions = {chunk.chunk_id: index for index, chunk in enumerate(retriever.chunks)}
        retriever.graph = {0: [1], 1: [0, 2], 2: [1]}
        retriever.index = faiss.IndexFlatIP(2)
        retriever.index.add(np.asarray([[1, 0], [0.8, 0.2], [0, 1]], dtype="float32"))

        seed = [(retriever.chunks[0], 0.9)]

        self.assertGreater(retriever._estimate_hop_gain(seed, hop=1), 0.0)
        self.assertEqual(retriever._estimate_hop_gain(seed, hop=1, excluded_indices={0, 1}), 0.0)

    def test_title_mention_creates_cross_document_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = SemanticRetriever(index_dir=Path(directory))
            retriever.chunks = [
                Chunk("0", "Arthur's Magazine", "", "Arthur's Magazine"),
                Chunk(
                    "1",
                    "First for Women",
                    "First for Women was published after Arthur's Magazine.",
                    "First for Women",
                ),
            ]
            graph_sets = {0: set(), 1: set()}

            bridge_count = retriever._add_title_bridge_edges(graph_sets)

        self.assertEqual(bridge_count, 1)
        self.assertEqual(graph_sets, {0: {1}, 1: {0}})

    def test_lowercase_title_span_is_still_recognised_as_bridge(self) -> None:
        aliases = {
            "arthur": [(("arthur", "s", "magazine"), "arthur s magazine")],
        }

        matches = _matched_title_aliases("It was published after arthur's magazine.", aliases)

        self.assertIn("arthur s magazine", matches)

    def test_graph_reranking_preserves_one_low_query_overlap_candidate(self) -> None:
        class FakeReranker:
            @staticmethod
            def predict(pairs, show_progress_bar=False):
                del pairs, show_progress_bar
                return np.asarray([5.0, 4.0, -5.0])

        retriever = SemanticRetriever(index_dir=Path("unused"))
        retriever.reranker = FakeReranker()
        direct_a = Chunk("direct-a", "source-a", "direct answer", "A")
        direct_b = Chunk("direct-b", "source-b", "another direct answer", "B")
        bridge = Chunk("bridge", "source-c", "second hop evidence", "C")
        initial = [(direct_a, 0.99), (direct_b, 0.98)]
        expanded = initial + [(bridge, 0.90)]

        result = retriever._rerank_graph_expansion("multi hop question", initial, expanded, top_k=2)

        self.assertEqual(len(result), 2)
        self.assertIn("bridge", {chunk.chunk_id for chunk, _ in result})


if __name__ == "__main__":
    unittest.main()
