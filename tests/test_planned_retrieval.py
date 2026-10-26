import unittest

from src.document_loader import Chunk
from src.planned_retrieval import StopDecision, run_planned_retrieval
from src.query_rewriter import QueryRewriteResult


class _Rewriter:
    def __init__(self):
        self.calls = 0

    def rewrite(self, question, evidence, previous_queries=()):
        del question, evidence, previous_queries
        self.calls += 1
        return QueryRewriteResult(
            query=f"query-{self.calls}",
            raw_response="",
            cache_hit=False,
            latency_seconds=0.0,
            prompt_eval_count=None,
            eval_count=None,
        )


class _Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=5):
        index = int(query.rsplit("-", 1)[1]) + 1
        return [(self.chunks[index], 0.95)][:top_k]


class _NewCandidateRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=5):
        step = int(query.rsplit("-", 1)[1])
        return [(self.chunks[4 + step], 0.95)][:top_k]


class _BatchedCandidateRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=5):
        step = int(query.rsplit("-", 1)[1])
        start = 3 + step * 2
        return [
            (self.chunks[index], 0.95 - offset * 0.01)
            for offset, index in enumerate(range(start, min(start + 2, len(self.chunks))))
        ][:top_k]


class _StopAfterTwo:
    def decide(self, state):
        return StopDecision(
            continue_search=state.search_step < 2,
            confidence=1.0,
            reason="test_stop",
        )


class PlannedRetrievalTests(unittest.TestCase):
    def test_each_hop_adds_one_candidate_and_preserves_previous_additions(self):
        chunks = [
            Chunk(str(index), f"source-{index}", f"text-{index}", f"title-{index}")
            for index in range(5)
        ]
        initial = [(chunks[0], 0.9), (chunks[1], 0.8)]
        result = run_planned_retrieval(
            _Retriever(chunks),
            _Rewriter(),
            "Question",
            initial,
            max_search_steps=3,
            top_k=5,
        )

        self.assertEqual(
            [chunk.chunk_id for chunk, _ in result.sources_by_step[3]],
            ["0", "1", "2", "3", "4"],
        )
        self.assertEqual([step.rewrite.query for step in result.steps], ["query-1", "query-2", "query-3"])
        self.assertFalse(hasattr(result.steps[0].state, "gold_hop"))

    def test_same_executor_can_enumerate_ten_steps(self):
        chunks = [
            Chunk(str(index), f"source-{index}", f"text-{index}", f"title-{index}")
            for index in range(12)
        ]
        result = run_planned_retrieval(
            _Retriever(chunks),
            _Rewriter(),
            "Question",
            [(chunks[0], 0.9), (chunks[1], 0.8)],
            max_search_steps=10,
            top_k=12,
        )

        self.assertEqual(len(result.steps), 10)
        self.assertEqual(max(result.sources_by_step), 10)
        self.assertEqual(len(result.memory), 12)

    def test_stop_policy_terminates_before_safety_cap(self):
        chunks = [
            Chunk(str(index), f"source-{index}", f"text-{index}", f"title-{index}")
            for index in range(12)
        ]
        result = run_planned_retrieval(
            _Retriever(chunks),
            _Rewriter(),
            "Question",
            [(chunks[0], 0.9), (chunks[1], 0.8)],
            max_search_steps=10,
            top_k=12,
            stop_policy=_StopAfterTwo(),
        )

        self.assertEqual(result.stop_step, 2)
        self.assertTrue(result.stopped_by_policy)
        self.assertEqual(len(result.steps), 2)

    def test_original_question_anchors_survive_later_rewrites(self):
        chunks = [
            Chunk(str(index), f"source-{index}", f"text-{index}", f"title-{index}")
            for index in range(8)
        ]
        initial = [(chunks[index], 0.9 - index * 0.1) for index in range(5)]
        result = run_planned_retrieval(
            _NewCandidateRetriever(chunks),
            _Rewriter(),
            "Question",
            initial,
            max_search_steps=3,
            top_k=5,
            anchor_slots=2,
        )

        final_ids = {chunk.chunk_id for chunk, _ in result.sources_by_step[3]}
        self.assertIn("0", final_ids)
        self.assertIn("1", final_ids)
        self.assertTrue({"5", "6", "7"}.issubset(final_ids))

    def test_evidence_survival_cap_accumulates_without_forgetting(self):
        chunks = [
            Chunk(str(index), f"source-{index}", f"text-{index}", f"title-{index}")
            for index in range(8)
        ]
        initial = [(chunks[index], 0.9 - index * 0.1) for index in range(5)]
        result = run_planned_retrieval(
            _NewCandidateRetriever(chunks),
            _Rewriter(),
            "Question",
            initial,
            max_search_steps=3,
            top_k=5,
            anchor_slots=2,
            reader_source_cap=8,
        )

        final_ids = {chunk.chunk_id for chunk, _ in result.sources_by_step[3]}
        self.assertEqual(final_ids, {str(index) for index in range(8)})

    def test_batched_additions_fill_cap_in_two_searches_then_freeze(self):
        chunks = [
            Chunk(str(index), f"source-{index}", f"text-{index}", f"title-{index}")
            for index in range(10)
        ]
        initial = [(chunks[index], 0.9 - index * 0.1) for index in range(5)]
        result = run_planned_retrieval(
            _BatchedCandidateRetriever(chunks),
            _Rewriter(),
            "Question",
            initial,
            max_search_steps=3,
            top_k=5,
            reader_source_cap=8,
            additions_per_step=2,
            replace_when_full=False,
        )

        self.assertEqual(len(result.sources_by_step[1]), 7)
        self.assertEqual(len(result.sources_by_step[2]), 8)
        self.assertEqual(result.sources_by_step[2], result.sources_by_step[3])
        self.assertEqual(len(result.steps[0].added_chunks), 2)
        self.assertEqual(len(result.steps[1].added_chunks), 1)
        self.assertEqual(len(result.steps[2].added_chunks), 0)


if __name__ == "__main__":
    unittest.main()
