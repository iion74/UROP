import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.evaluate_answer_oracle import (
    SEQUENTIAL_STATE_FEATURE_NAMES,
    _best_answer_action,
    _best_state_action,
    _evaluate_example,
    _evaluate_sequential_trajectories,
    _sequential_state_features,
)
from experiments.train_multihop_policy import CorpusDocument, MultiHopExample
from src.document_loader import Chunk
from src.qa_evaluation import (
    EVIDENCE_CHAIN_PROMPT_VERSION,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    ReaderResult,
    answer_score,
    clean_prediction,
    evidence_context,
    normalize_answer,
    ollama_model_identity,
    token_f1,
)
from src.routing import POLICY_ACTION_IDS, EvidenceProfile, QueryRouter


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.post_calls = 0
        self.last_request = None

    def get(self, url, timeout):
        del url, timeout
        return FakeResponse({"models": [{"name": "qwen3:8b", "digest": "sha256:test"}]})

    def post(self, url, json, timeout):
        del url, timeout
        self.post_calls += 1
        self.last_request = json
        return FakeResponse(
            {
                "response": "Answer: Arthur's Magazine",
                "prompt_eval_count": 100,
                "eval_count": 4,
            }
        )


class QAEvaluationTests(unittest.TestCase):
    def test_standard_answer_normalization(self) -> None:
        self.assertEqual(normalize_answer("The, Arthur's Magazine!"), "arthurs magazine")
        self.assertEqual(normalize_answer("  An   Example "), "example")

    def test_answer_score_uses_best_alias(self) -> None:
        score = answer_score("NYC", "New York City", ["NYC"])

        self.assertEqual(score.em, 1.0)
        self.assertEqual(score.f1, 1.0)
        self.assertEqual(score.matched_gold, "NYC")

    def test_special_yes_no_answers_require_exact_class(self) -> None:
        self.assertEqual(token_f1("yes definitely", "yes"), 0.0)
        self.assertEqual(token_f1("no", "yes"), 0.0)

    def test_musique_uses_squad_partial_credit_for_yes_no(self) -> None:
        score = answer_score("yes definitely", "yes", dataset="musique")

        self.assertAlmostEqual(score.f1, 2 / 3)

    def test_hotpot_empty_normalized_f1_matches_official_zero(self) -> None:
        self.assertEqual(token_f1("the", "a"), 0.0)

    def test_2wiki_alias_can_receive_exact_match(self) -> None:
        score = answer_score("Canadian", "Canada", ("Canadian",), dataset="2wiki")

        self.assertEqual((score.em, score.f1), (1.0, 1.0))

    def test_numeric_and_date_punctuation_follow_official_normalization(self) -> None:
        self.assertEqual(answer_score("1,960", "1960").em, 1.0)
        self.assertEqual(answer_score("May 4, 1960", "May 4 1960").f1, 1.0)

    def test_prediction_cleanup_removes_answer_wrapper(self) -> None:
        self.assertEqual(clean_prediction("Answer: Paris\nBecause the evidence says so."), "Paris")
        self.assertEqual(clean_prediction('{"answer": "1960"}'), "1960")

    def test_evidence_context_obeys_one_shared_character_budget(self) -> None:
        sources = [
            (Chunk("c1", "d1", "A" * 100, "First"), 0.9),
            (Chunk("c2", "d2", "B" * 100, "Second"), 0.8),
        ]

        context, provenance = evidence_context(sources, 50)

        self.assertLessEqual(len(context), 50)
        self.assertEqual(len(provenance), 1)
        self.assertEqual(provenance[0]["content_id"], "d1")

    def test_balanced_evidence_context_keeps_later_sources_visible(self) -> None:
        sources = [
            (Chunk(f"c{index}", f"d{index}", letter * 100, f"Doc {index}"), 1.0 / index)
            for index, letter in enumerate("ABCDEFGH", start=1)
        ]

        context, provenance = evidence_context(sources, 400, allocation="balanced")

        self.assertLessEqual(len(context), 400)
        self.assertEqual([item["content_id"] for item in provenance], [f"d{i}" for i in range(1, 9)])
        self.assertIn("[1] Doc 1", context)
        self.assertIn("[8] Doc 8", context)
        excerpts = [section.split("\n", 1)[1] for section in context.split("\n\n")]
        self.assertLessEqual(max(map(len, excerpts)) - min(map(len, excerpts)), 1)

    def test_balanced_evidence_context_redistributes_short_source_budget(self) -> None:
        sources = [
            (Chunk("c1", "d1", "A", "Short"), 0.9),
            (Chunk("c2", "d2", "B" * 100, "Long"), 0.8),
        ]

        context, provenance = evidence_context(sources, 80, allocation="balanced")

        self.assertLessEqual(len(context), 80)
        self.assertEqual(len(provenance), 2)
        first, second = [section.split("\n", 1)[1] for section in context.split("\n\n")]
        self.assertEqual(first, "A")
        self.assertGreater(len(second), len(first))

    def test_query_focused_context_keeps_relevant_sentences_in_original_order(self) -> None:
        source = (
            Chunk(
                "c1",
                "d1",
                "Noise about apples. Jefferson Davis fought until 1848. "
                "More unrelated material. The Mexican-American War ended in 1848.",
                "Jefferson Davis",
            ),
            0.9,
        )

        context, provenance = evidence_context(
            [source],
            500,
            compression="query_focused",
            focus_queries=("When did Jefferson Davis leave the Mexican-American War?",),
            focused_sentences_per_source=2,
        )

        self.assertEqual(len(provenance), 1)
        self.assertIn("Jefferson Davis fought until 1848.", context)
        self.assertIn("The Mexican-American War ended in 1848.", context)
        self.assertNotIn("Noise about apples.", context)

    def test_frozen_reader_caches_identical_prompt(self) -> None:
        session = FakeSession()
        with TemporaryDirectory() as directory:
            with AnswerCache(Path(directory) / "answers.sqlite3") as cache:
                reader = FrozenOllamaReader(
                    ReaderConfig(
                        host="http://localhost:11434",
                        model="qwen3:8b",
                        model_digest="sha256:test",
                    ),
                    cache,
                    session=session,
                )

                first = reader.answer("Which?", "Evidence")
                second = reader.answer("Which?", "Evidence")

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.prediction, "Arthur's Magazine")
        self.assertEqual(session.post_calls, 1)
        self.assertEqual(session.last_request["format"]["required"], ["answer"])
        self.assertEqual(session.last_request["options"]["temperature"], 0.0)
        self.assertFalse(session.last_request["think"])

    def test_frozen_reader_can_enable_internal_reasoning(self) -> None:
        session = FakeSession()
        with TemporaryDirectory() as directory:
            with AnswerCache(Path(directory) / "answers.sqlite3") as cache:
                reader = FrozenOllamaReader(
                    ReaderConfig(
                        host="http://localhost:11434",
                        model="qwen3:8b",
                        model_digest="sha256:test",
                        thinking=True,
                    ),
                    cache,
                    session=session,
                )
                reader.answer("Which?", "Evidence")

        self.assertTrue(session.last_request["think"])

    def test_frozen_reader_can_require_structured_evidence_chain(self) -> None:
        session = FakeSession()
        with TemporaryDirectory() as directory:
            with AnswerCache(Path(directory) / "answers.sqlite3") as cache:
                reader = FrozenOllamaReader(
                    ReaderConfig(
                        host="http://localhost:11434",
                        model="qwen3:8b",
                        model_digest="sha256:test",
                        prompt_version=EVIDENCE_CHAIN_PROMPT_VERSION,
                    ),
                    cache,
                    session=session,
                )
                reader.answer("Which?", "Evidence")

        self.assertEqual(
            session.last_request["format"]["required"],
            ["evidence_chain", "answer"],
        )
        self.assertIn("shortest chain", session.last_request["prompt"])

    def test_ollama_identity_records_digest(self) -> None:
        name, digest = ollama_model_identity(
            "http://localhost:11434",
            "qwen3:8b",
            session=FakeSession(),
        )

        self.assertEqual(name, "qwen3:8b")
        self.assertEqual(digest, "sha256:test")

    def test_answer_oracle_breaks_f1_ties_by_cost(self) -> None:
        actions = {
            "vector": {"f1": 1.0, "em": 1.0, "extra_candidates": 0},
            "graph_1hop_small": {"f1": 1.0, "em": 1.0, "extra_candidates": 3},
        }

        self.assertEqual(_best_answer_action(actions), "vector")

    def test_sequential_state_records_observed_change_and_budget(self) -> None:
        initial_profile = EvidenceProfile(0.8, 0.7, 0.2, 0.1, 0.6, 0.2, 0.1)
        current_profile = EvidenceProfile(0.6, 0.5, 0.5, 0.4, 0.3, 0.7, 0.6)
        first = Chunk("c1", "d1", "First", "First")
        second = Chunk("c2", "d2", "Second", "Second")

        features = _sequential_state_features(
            initial_profile,
            current_profile,
            [(first, 0.9)],
            [(second, 0.8)],
            first_action="graph_1hop_small",
            first_extra_candidates=3,
        )

        self.assertEqual(len(features), len(SEQUENTIAL_STATE_FEATURE_NAMES))
        self.assertAlmostEqual(features[8], -0.2)
        self.assertEqual(features[-3:-1].tolist(), [1.0, 0.0])
        self.assertEqual(features[-1], 0.5)

    def test_sequential_action_tie_prefers_stop_and_lower_cost(self) -> None:
        trajectories = {"stop": "t0", "expand_2hop_small": "t1"}
        results = {
            "t0": {"f1": 1.0, "em": 1.0, "extra_candidates": 3},
            "t1": {"f1": 1.0, "em": 1.0, "extra_candidates": 6},
        }

        self.assertEqual(_best_state_action(trajectories, results), "stop")

    def test_sequential_reuses_same_two_hop_terminal_as_one_shot(self) -> None:
        direct = (Chunk("direct", "d0", "Direct", "Direct"), 0.9)
        first_small = (Chunk("small", "d1", "Small", "Small"), 0.8)
        first_large = (Chunk("large", "d2", "Large", "Large"), 0.7)

        class RetrieverStub:
            @staticmethod
            def _evidence_profile(sources):
                return EvidenceProfile(
                    0.5,
                    0.5,
                    min(1.0, len(sources) / 5),
                    0.2,
                    0.1,
                    0.3,
                    0.1,
                )

        def result(f1, extra, context_hash):
            return {
                "f1": f1,
                "em": f1,
                "extra_candidates": extra,
                "action_retrieval_latency_seconds": 0.1,
                "source_recall": f1,
                "context_hash": context_hash,
            }

        action_results = {
            "vector": result(0.0, 0, "direct"),
            "graph_1hop_small": result(0.0, 2, "one-small"),
            "graph_1hop_large": result(0.0, 4, "one-large"),
            "graph_2hop_small": result(1.0, 4, "two-small"),
            "graph_2hop_large": result(0.5, 6, "two-large"),
        }

        sequential = _evaluate_sequential_trajectories(
            RetrieverStub(),
            initial_profile=RetrieverStub._evidence_profile([direct]),
            initial_sources=[direct],
            first_sources={
                "graph_1hop_small": [first_small],
                "graph_1hop_large": [first_large],
            },
            action_results=action_results,
        )

        path = "graph_1hop_small_expand_2hop_small"
        self.assertEqual(sequential["oracle_f1"], 1.0)
        self.assertTrue(sequential["shared_terminal_oracle"])
        self.assertEqual(sequential["second_step_results"][path]["context_hash"], "two-small")
        self.assertEqual(sequential["second_step_results"][path]["extra_candidates"], 6)

    def test_example_evaluation_records_answer_and_evidence_metrics(self) -> None:
        document = CorpusDocument("content:gold", "Paris", "Paris is the capital of France.")
        example = MultiHopExample(
            dataset="hotpotqa",
            example_id="q1",
            question="What is the capital of France?",
            documents=(document,),
            supporting_groups=(frozenset({document.content_id}),),
            hop_count=2,
            reasoning_type="bridge",
            answer="Paris",
        )
        source = (Chunk("chunk:1", document.content_id, document.text, document.title), 0.9)

        class RetrieverStub:
            chunks = [source[0]]

            @staticmethod
            def search(question, top_k):
                del question, top_k
                return [source]

            @staticmethod
            def _rerank(question, candidates):
                del question
                return candidates

            @staticmethod
            def _evidence_profile(initial):
                del initial
                return EvidenceProfile(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        class ReaderStub:
            @staticmethod
            def answer(question, context):
                del question, context
                return ReaderResult("Paris", "Paris", False, 0.1, 10, 1)

        with patch(
            "experiments.evaluate_answer_oracle.retrieve_for_action",
            return_value=([source], 0),
        ):
            row = _evaluate_example(
                RetrieverStub(),
                ReaderStub(),
                example,
                actions=POLICY_ACTION_IDS,
                top_k=5,
                context_character_budget=1000,
            )

        self.assertEqual(row["direct_f1"], 1.0)
        self.assertEqual(row["oracle_action"], "vector")
        self.assertEqual(row["actions"]["vector"]["all_support_recall"], 1.0)
        self.assertEqual(row["gold_support_f1"], 1.0)

    def test_sequential_example_enumerates_post_observation_trajectories(self) -> None:
        document = CorpusDocument("content:gold", "Paris", "Paris is the capital of France.")
        example = MultiHopExample(
            dataset="hotpotqa",
            example_id="q-sequential",
            question="What is the capital of France?",
            documents=(document,),
            supporting_groups=(frozenset({document.content_id}),),
            hop_count=2,
            reasoning_type="bridge",
            answer="Paris",
        )
        source = (Chunk("chunk:1", document.content_id, document.text, document.title), 0.9)

        class RetrieverStub:
            chunks = [source[0]]
            chunk_positions = {source[0].chunk_id: 0}
            rule_router = QueryRouter()

            @staticmethod
            def search(question, top_k):
                del question, top_k
                return [source]

            @staticmethod
            def _evidence_profile(initial):
                del initial
                return EvidenceProfile(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)

            @staticmethod
            def _expand_for_plan(initial, decision):
                del decision
                return initial

            @staticmethod
            def _graph_candidates(seeds, max_hops, excluded_indices):
                del seeds, max_hops, excluded_indices
                return {}

            @staticmethod
            def _select_expansion_candidates(candidates, anchors, budget):
                del candidates, anchors, budget
                return []

            @staticmethod
            def _rerank_graph_expansion(question, initial, expanded, top_k):
                del question, initial
                return expanded[:top_k]

            @staticmethod
            def _rerank(question, candidates):
                del question
                return candidates

        class ReaderStub:
            @staticmethod
            def answer(question, context):
                del question, context
                return ReaderResult("Paris", "Paris", True, 0.0, 10, 1)

        with patch(
            "experiments.evaluate_answer_oracle.retrieve_for_action",
            return_value=([source], 0),
        ):
            row = _evaluate_example(
                RetrieverStub(),
                ReaderStub(),
                example,
                actions=POLICY_ACTION_IDS,
                top_k=5,
                context_character_budget=1000,
                sequential_trajectories=True,
            )

        self.assertEqual(len(row["sequential"]["terminal_results"]), 9)
        self.assertEqual(len(row["sequential"]["second_step_results"]), 4)
        self.assertEqual(row["sequential"]["s0_oracle_action"], "stop")
        self.assertEqual(row["sequential_oracle_trajectory"], "stop")
        self.assertEqual(row["sequential_oracle_f1"], row["oracle_f1"])


if __name__ == "__main__":
    unittest.main()
