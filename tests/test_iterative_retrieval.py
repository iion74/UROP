import unittest

import numpy as np

from src.document_loader import Chunk
from src.iterative_retrieval import (
    BridgeSearchConfig,
    IterativeSearchConfig,
    compose_bridge_evidence,
    compose_iterative_evidence,
    evidence_chain_query,
    rank_bridge_paths,
    rank_iterative_chains,
    title_is_mentioned,
)


class _RerankerStub:
    def predict(self, pairs, show_progress_bar=False):
        del show_progress_bar
        return np.asarray(
            [
                6.0
                if "Fourth Evidence" in candidate and "Third Evidence" in query
                else 6.0
                if "Third Evidence" in candidate and "Correct Bridge:" in query
                else 6.0
                if "Correct Bridge" in candidate and "Third Evidence" not in query
                else -1.0
                for query, candidate in pairs
            ],
            dtype="float64",
        )


class _RetrieverStub:
    def __init__(self) -> None:
        self.chunks = [
            Chunk("a", "source-a", "The director was Correct Bridge.", "First Film"),
            Chunk("d", "source-d", "A direct distractor.", "Distractor"),
            Chunk("b", "source-b", "Correct Bridge was born in Seoul.", "Correct Bridge"),
            Chunk("n", "source-n", "An unrelated semantic neighbour.", "Noise"),
            Chunk("c", "source-c", "Third Evidence points to Fourth Evidence.", "Third Evidence"),
            Chunk("e", "source-e", "The final fact is here.", "Fourth Evidence"),
        ]
        self.chunk_positions = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
        self.graph = {0: [2, 3], 1: [], 2: [0, 4], 3: [0], 4: [2, 5], 5: [4]}

    def _reachable_distances(self, start, max_hops):
        distances = {start: 0}
        frontier = [start]
        while frontier:
            node = frontier.pop(0)
            if distances[node] >= max_hops:
                continue
            for neighbour in self.graph.get(node, []):
                if neighbour not in distances:
                    distances[neighbour] = distances[node] + 1
                    frontier.append(neighbour)
        return distances

    @staticmethod
    def _similarity(first, second):
        return 0.2 if second == 2 else 0.8

    @staticmethod
    def _reranker():
        return _RerankerStub()

    def search(self, query, top_k=5):
        del query
        scores = {"b": 0.95, "c": 0.90, "e": 0.85, "n": 0.20, "d": 0.10, "a": 0.05}
        ranked = sorted(self.chunks, key=lambda chunk: scores[chunk.chunk_id], reverse=True)
        return [(chunk, scores[chunk.chunk_id]) for chunk in ranked[:top_k]]


class IterativeRetrievalTests(unittest.TestCase):
    def test_title_match_handles_parenthetical_alias(self) -> None:
        self.assertTrue(title_is_mentioned("Correct Bridge (person)", "The director was Correct Bridge."))
        self.assertFalse(title_is_mentioned("Bridge", "The word bridgework is different."))

    def test_evidence_conditioned_score_prefers_true_title_bridge(self) -> None:
        retriever = _RetrieverStub()
        initial = [(retriever.chunks[0], 0.9), (retriever.chunks[1], 0.8)]

        result = rank_bridge_paths(
            retriever,
            "Where was the director born?",
            initial,
            BridgeSearchConfig(seed_count=1, candidates_per_seed=4),
        )

        self.assertEqual(result.evaluated_pairs, 2)
        self.assertEqual(result.paths[0].candidate.chunk_id, "b")
        self.assertTrue(result.paths[0].title_mentioned)

    def test_composition_keeps_complete_path_then_direct_fill(self) -> None:
        retriever = _RetrieverStub()
        initial = [(retriever.chunks[0], 0.9), (retriever.chunks[1], 0.8)]
        result = rank_bridge_paths(
            retriever,
            "Where was the director born?",
            initial,
            BridgeSearchConfig(seed_count=1, candidates_per_seed=4),
        )

        sources = compose_bridge_evidence(initial, result.paths, top_k=3, max_paths=1)

        self.assertEqual([chunk.chunk_id for chunk, _ in sources], ["a", "b", "d"])
        self.assertEqual(len({chunk.chunk_id for chunk, _ in sources}), 3)

    def test_no_path_falls_back_to_direct(self) -> None:
        retriever = _RetrieverStub()
        retriever.graph = {}
        initial = [(retriever.chunks[0], 0.9), (retriever.chunks[1], 0.8)]

        result = rank_bridge_paths(
            retriever,
            "Question",
            initial,
            BridgeSearchConfig(),
        )
        sources = compose_bridge_evidence(initial, result.paths, top_k=1, max_paths=1)

        self.assertFalse(result.paths)
        self.assertEqual(sources, initial[:1])

    def test_iterative_beam_accumulates_three_and_four_hop_chains(self) -> None:
        retriever = _RetrieverStub()
        initial = [(retriever.chunks[0], 0.9), (retriever.chunks[1], 0.8)]

        result = rank_iterative_chains(
            retriever,
            "What is the final fact?",
            initial,
            BridgeSearchConfig(seed_count=1, candidates_per_seed=4),
            IterativeSearchConfig(max_evidence_hops=4, beam_width=3, candidates_per_state=4),
        )

        self.assertEqual([len(result.chains_by_hop[hop][0].chunks) for hop in (2, 3, 4)], [2, 3, 4])
        sources = compose_iterative_evidence(
            initial,
            result.chains_by_hop,
            evidence_hops=4,
            top_k=5,
        )
        self.assertEqual([chunk.chunk_id for chunk, _ in sources], ["a", "d", "b", "c", "e"])

    def test_chain_query_contains_all_previous_evidence_titles(self) -> None:
        retriever = _RetrieverStub()
        query = evidence_chain_query(
            "Question",
            (retriever.chunks[0], retriever.chunks[2], retriever.chunks[4]),
            max_characters=600,
        )

        self.assertIn("First Film", query)
        self.assertIn("Correct Bridge", query)
        self.assertIn("Third Evidence", query)


if __name__ == "__main__":
    unittest.main()
