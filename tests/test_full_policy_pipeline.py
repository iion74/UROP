import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.evaluate_context_answer_oracle import _compact_row
from experiments.train_full_gated_policy import (
    _best_static_action,
    _effective_dataset_totals,
    _fit_final,
    _nested_audit,
    _sample_weights,
)
from src.policy import POLICY_FEATURE_NAMES
from src.document_loader import Chunk
from src.retriever import SemanticRetriever


class _EmbeddingStub:
    @staticmethod
    def encode(texts, normalize_embeddings, show_progress_bar=False):
        del normalize_embeddings, show_progress_bar
        vectors = np.zeros((len(texts), 3), dtype="float32")
        for index in range(len(texts)):
            vectors[index, index % 3] = 1.0
        return vectors


class FullPolicyPipelineTests(unittest.TestCase):
    def test_in_memory_retriever_build_does_not_write_index(self) -> None:
        with TemporaryDirectory() as directory:
            index_dir = Path(directory) / "should_not_exist"
            retriever = SemanticRetriever(index_dir)
            retriever.model = _EmbeddingStub()
            retriever.build(
                [
                    Chunk("c1", "d1", "Alpha mentions Beta.", "Alpha"),
                    Chunk("c2", "d2", "Beta evidence.", "Beta"),
                ],
                persist=False,
                semantic_neighbors=1,
            )

            self.assertTrue(retriever.ready)
            self.assertFalse(index_dir.exists())

    def test_compact_oracle_row_keeps_scores_but_drops_raw_generation(self) -> None:
        action = {
            "prediction": "Paris",
            "raw_response": "a very large raw response",
            "matched_gold": "Paris",
            "em": 1.0,
            "f1": 1.0,
            "source_recall": 1.0,
            "all_support_recall": 1.0,
            "extra_candidates": 0,
            "reader_cache_hit": False,
            "evidence": [{"content_id": "doc-1", "text": "large evidence"}],
        }
        row = {
            "id": "hotpotqa:1",
            "actions": {"vector": action},
            "direct_f1": 1.0,
            "oracle_f1": 1.0,
        }

        compact = _compact_row(row)

        self.assertEqual(compact["actions"]["vector"]["f1"], 1.0)
        self.assertEqual(compact["actions"]["vector"]["evidence_content_ids"], ["doc-1"])
        self.assertNotIn("raw_response", compact["actions"]["vector"])
        self.assertNotIn("evidence", compact["actions"]["vector"])

    def test_temperature_weights_use_all_rows_and_balance_dataset_totals(self) -> None:
        rows = [
            *[{"dataset": "hotpotqa"} for _ in range(2)],
            *[{"dataset": "musique"} for _ in range(4)],
            *[{"dataset": "2wiki"} for _ in range(8)],
        ]

        balanced = _effective_dataset_totals(rows, alpha=0.0)
        proportional = _effective_dataset_totals(rows, alpha=1.0)

        self.assertEqual(len(_sample_weights(rows, 0.0)), len(rows))
        self.assertAlmostEqual(balanced["hotpotqa"], balanced["musique"])
        self.assertAlmostEqual(balanced["musique"], balanced["2wiki"])
        self.assertAlmostEqual(proportional["musique"] / proportional["hotpotqa"], 2.0)
        self.assertAlmostEqual(proportional["2wiki"] / proportional["hotpotqa"], 4.0)

    def test_static_baseline_is_selected_by_dataset_macro_f1(self) -> None:
        rows = []
        for dataset, small, large in (("a", 0.9, 0.8), ("b", 0.1, 0.8)):
            rows.append(
                {
                    "dataset": dataset,
                    "actions": {
                        "graph_1hop_small": {"f1": small},
                        "graph_1hop_large": {"f1": large},
                        "graph_2hop_small": {"f1": 0.0},
                        "graph_2hop_large": {"f1": 0.0},
                    },
                }
            )

        self.assertEqual(_best_static_action(rows, ("a", "b")), "graph_1hop_large")

    def test_nested_full_trainer_produces_one_oof_record_per_row(self) -> None:
        rows = []
        datasets = ("hotpotqa", "musique", "2wiki")
        for dataset in datasets:
            for index in range(30):
                positive = index % 2 == 0
                direct = 0.0 if positive else 1.0
                graph = 1.0 if positive else 0.0
                rows.append(
                    {
                        "id": f"{dataset}:{index}",
                        "dataset": dataset,
                        "question": "bridge relation positive" if positive else "direct fact negative",
                        "features": {
                            name: (0.8 if positive else 0.2) + feature_index * 0.001
                            for feature_index, name in enumerate(POLICY_FEATURE_NAMES)
                        },
                        "direct_em": direct,
                        "direct_f1": direct,
                        "oracle_em": 1.0,
                        "oracle_f1": 1.0,
                        "actions": {
                            "vector": {"em": direct, "f1": direct},
                            "graph_1hop_small": {"em": graph, "f1": graph},
                            "graph_1hop_large": {"em": graph, "f1": graph},
                            "graph_2hop_small": {"em": graph, "f1": graph},
                            "graph_2hop_large": {"em": graph, "f1": graph},
                        },
                    }
                )
        args = Namespace(
            datasets=datasets,
            mixing_alpha=0.0,
            seed=42,
            max_text_features=100,
            trees=4,
            min_samples_leaf=1,
            max_features=1.0,
            folds=3,
            calibration_ratio=0.25,
            dataset_harm_penalty=0.0,
        )
        grid = np.asarray([0.25, 0.5, 0.75])

        records, gains = _nested_audit(rows, args, grid)
        models, threshold = _fit_final(rows, args, grid)

        self.assertEqual(len(records), len(rows))
        self.assertEqual(len({record["id"] for record in records}), len(rows))
        self.assertEqual(len(gains), args.folds)
        self.assertTrue(0.0 <= threshold <= 1.0)
        self.assertEqual(models.value.predict(np.zeros((1, len(POLICY_FEATURE_NAMES)))).shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
