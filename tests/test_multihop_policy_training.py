import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from experiments.train_multihop_policy import (
    CorpusDocument,
    MultiHopExample,
    _best_action,
    _corpus_fingerprint,
    _load_or_label_actions,
    _normalise_processed_record,
    _reservoir_sample,
    _training_weights,
    evidence_metrics,
    select_query_examples,
)
from src.routing import POLICY_ACTION_IDS
from src.routing import EvidenceProfile


def _actions(vector_utility=0.5, graph_utility=0.4):
    return {
        action: {
            "utility": vector_utility if action == "vector" else graph_utility,
            "source_recall": 0.5 if action == "vector" else 1.0,
            "all_support_recall": 0.0 if action == "vector" else 1.0,
            "extra_candidates": 0 if action == "vector" else 3,
        }
        for action in POLICY_ACTION_IDS
    }


def _example(example_id: str, *documents: CorpusDocument) -> MultiHopExample:
    return MultiHopExample(
        dataset="hotpotqa",
        example_id=example_id,
        question=f"Question {example_id}?",
        documents=tuple(documents),
        supporting_groups=(frozenset({documents[0].content_id}),),
        hop_count=2,
        reasoning_type="bridge",
    )


class MultiHopPolicyTrainingTests(unittest.TestCase):
    def test_processed_record_maps_local_gold_to_content_identity(self) -> None:
        record = {
            "dataset": "musique",
            "id": "m1",
            "question": "Question?",
            "hop_count": 2,
            "reasoning_type": "2hop",
            "documents": [
                {"doc_id": "m1:p1", "content_id": "content:a", "title": "Same", "text": "Wrong."},
                {"doc_id": "m1:p2", "content_id": "content:b", "title": "Same", "text": "Gold."},
            ],
            "gold_supports": [{"doc_ids": ["m1:p2"]}],
        }

        example = _normalise_processed_record(record)

        self.assertIsNotNone(example)
        self.assertEqual(example.supporting_groups, (frozenset({"content:b"}),))
        self.assertEqual({document.content_id for document in example.documents}, {"content:a", "content:b"})

    def test_all_support_requires_every_gold_group(self) -> None:
        groups = (frozenset({"a", "a-alias"}), frozenset({"b"}))

        self.assertEqual(evidence_metrics({"a"}, groups), (0.5, 0.0))
        self.assertEqual(evidence_metrics({"a-alias", "b"}, groups), (1.0, 1.0))

    def test_best_action_prioritises_joint_evidence_utility(self) -> None:
        actions = _actions(vector_utility=0.15, graph_utility=0.80)

        self.assertEqual(_best_action(actions), "graph_1hop_small")

    def test_dataset_balancing_equalises_total_weight(self) -> None:
        rows = [
            {"dataset": "hotpotqa", "actions": _actions()},
            {"dataset": "hotpotqa", "actions": _actions()},
            {"dataset": "2wiki", "actions": _actions()},
        ]

        weights = _training_weights(rows, graph_positive_weight=8.0, balance_datasets=True)

        self.assertTrue(np.allclose(weights, [0.75, 0.75, 1.5]))
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2:].sum()))

    def test_graph_advantage_multiplies_sample_weight(self) -> None:
        rows = [{"dataset": "musique", "actions": _actions(vector_utility=0.3, graph_utility=0.8)}]

        weights = _training_weights(rows, graph_positive_weight=8.0, balance_datasets=True)

        self.assertEqual(weights.tolist(), [8.0])

    def test_reservoir_sample_is_reproducible(self) -> None:
        records = [{"id": index} for index in range(100)]

        first = _reservoir_sample(iter(records), 10, 42)
        second = _reservoir_sample(iter(records), 10, 42)

        self.assertEqual(first, second)
        self.assertNotEqual({row["id"] for row in first}, set(range(10)))

    def test_query_cohorts_are_reproducible_and_nested(self) -> None:
        document = CorpusDocument("content:shared", "Shared", "Shared text.")
        examples = [_example(str(index), document) for index in range(20)]

        small = select_query_examples(reversed(examples), 5, 42)
        large = select_query_examples(examples, 10, 42)
        repeated = select_query_examples(examples, 5, 42)

        self.assertEqual([row.policy_id for row in small], [row.policy_id for row in repeated])
        self.assertLessEqual({row.policy_id for row in small}, {row.policy_id for row in large})

    def test_corpus_fingerprint_depends_on_documents_not_query_cohort(self) -> None:
        shared = CorpusDocument("content:shared", "Shared", "Shared text.")
        first = _example("q1", shared)
        second = _example("q2", shared)

        self.assertEqual(_corpus_fingerprint([first]), _corpus_fingerprint([first, second]))

        changed = CorpusDocument("content:shared", "Shared", "Changed text.")
        self.assertNotEqual(_corpus_fingerprint([first]), _corpus_fingerprint([_example("q1", changed)]))

    def test_action_row_cache_expands_with_nested_query_cohort(self) -> None:
        document = CorpusDocument("content:shared", "Shared", "Shared text.")
        first = _example("q1", document)
        second = _example("q2", document)

        class RetrieverStub:
            @staticmethod
            def search(question, top_k):
                del question, top_k
                return []

            @staticmethod
            def _evidence_profile(initial):
                del initial
                return EvidenceProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        config = {"format_version": 3, "corpus_fingerprint": "fixed"}
        with TemporaryDirectory() as directory:
            rows_path = Path(directory) / "rows.jsonl"
            with patch(
                "experiments.train_multihop_policy._evaluate_actions",
                return_value=_actions(),
            ) as evaluate:
                _load_or_label_actions(
                    RetrieverStub(),
                    [first],
                    rows_path,
                    config,
                    top_k=5,
                    penalty=0.02,
                    joint_weight=0.7,
                    log_every=1,
                    relabel=False,
                )
                self.assertEqual(evaluate.call_count, 1)

                rows = _load_or_label_actions(
                    RetrieverStub(),
                    [first, second],
                    rows_path,
                    config,
                    top_k=5,
                    penalty=0.02,
                    joint_weight=0.7,
                    log_every=1,
                    relabel=False,
                )
                self.assertEqual(evaluate.call_count, 2)
                self.assertEqual([row["id"] for row in rows], [first.policy_id, second.policy_id])

                _load_or_label_actions(
                    RetrieverStub(),
                    [first],
                    rows_path,
                    config,
                    top_k=5,
                    penalty=0.02,
                    joint_weight=0.7,
                    log_every=1,
                    relabel=False,
                )
                self.assertEqual(evaluate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
