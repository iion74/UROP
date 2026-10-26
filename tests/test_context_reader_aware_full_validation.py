import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.analyze_context_reader_aware_multiseed import _estimated_latencies
from experiments.analyze_student_policy_multiseed import _costs
from experiments.evaluate_context_reader_aware_prefixes import (
    CONTEXT_PREFIX_FEATURE_NAMES,
    _retrieval_features,
    _sanitize_unicode,
)
from experiments.train_adaptive_proof_completion_policy import ACTIONS
from experiments.train_reader_aware_sufficiency_policy import _load
from experiments.train_reader_free_student_policy import policy_features, student_features
from src.document_loader import Chunk


class ContextReaderAwareFullValidationTests(unittest.TestCase):
    def test_lone_unicode_surrogates_are_replaced_before_jsonl_write(self) -> None:
        value = {"prediction": "answer \ud83d", "chain": ["정상", "bad \udc00"]}
        sanitized = _sanitize_unicode(value)
        encoded = json.dumps(sanitized, ensure_ascii=False).encode("utf-8")
        self.assertIn(b"answer ?", encoded)
        self.assertIn(b"bad ?", encoded)

    def test_context_retrieval_features_are_observable_and_complete(self) -> None:
        sources = [
            (Chunk("c1", "d1", "Alpha text", "Alpha"), 0.9),
            (Chunk("c2", "d2", "Beta text", "Beta"), 0.6),
        ]
        planner_step = SimpleNamespace(
            rewrite=SimpleNamespace(sufficient=False, missing_information="Beta birth place"),
            state=SimpleNamespace(consecutive_no_new=0),
            added_chunks=(sources[1][0],),
        )
        values = _retrieval_features(2, sources, {"d1"}, planner_step)
        self.assertEqual(len(values), len(CONTEXT_PREFIX_FEATURE_NAMES) - 19)
        by_name = dict(
            zip(CONTEXT_PREFIX_FEATURE_NAMES[: len(values)], values, strict=True)
        )
        self.assertEqual(by_name["search_step"], 2.0)
        self.assertEqual(by_name["new_candidate_fraction"], 0.5)
        self.assertEqual(by_name["previous_candidate_survival_fraction"], 1.0)
        self.assertEqual(by_name["planner_sufficient_known"], 1.0)

    def test_seeded_train_pool_sampling_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in ("hotpotqa", "musique", "2wiki"):
                path = root / f"{dataset}_train.jsonl"
                with path.open("w", encoding="utf-8") as destination:
                    for index in range(30):
                        destination.write(
                            json.dumps({"id": f"{dataset}:{index}", "dataset": dataset}) + "\n"
                        )
                path.with_suffix(".manifest.json").write_text(
                    json.dumps({"status": "ready", "feature_names": ["feature"]}),
                    encoding="utf-8",
                )
            first = _load(root, ("hotpotqa", "musique", "2wiki"), "train", 10, sample_seed=42)
            repeated = _load(root, ("hotpotqa", "musique", "2wiki"), "train", 10, sample_seed=42)
            different = _load(root, ("hotpotqa", "musique", "2wiki"), "train", 10, sample_seed=43)
            self.assertEqual([row["id"] for row in first], [row["id"] for row in repeated])
            self.assertNotEqual({row["id"] for row in first}, {row["id"] for row in different})
            self.assertEqual(len(first), 30)

    def test_online_latency_counts_prefix_readers_for_adaptive_policy(self) -> None:
        actions = {}
        for step, action in enumerate(ACTIONS):
            actions[action] = {
                "reader_latency_seconds": 1.0,
                "planner_rewrite_latency_cumulative_seconds": 1.2 * step,
            }
        prefix = {
            "initial_retrieval_latency_seconds": 0.5,
            "planner_total_latency_seconds": 3.0,
            "actions": actions,
        }
        direct, fixed3, adaptive_step2 = _estimated_latencies(prefix, 2)
        self.assertEqual(direct, 1.5)
        self.assertEqual(fixed3, 4.5)
        self.assertEqual(adaptive_step2, 5.9)

    def test_reader_free_student_features_ignore_reader_outputs(self) -> None:
        manifest_names = ("search_step", "retrieval_score", "answer_is_unknown")
        actions = {}
        for step, action in enumerate(ACTIONS[:3]):
            actions[action] = {
                "policy_features": [float(step), 0.8 - step * 0.1, float(step % 2)],
                "candidate_evidence": [
                    {"content_id": f"d{step}", "title": "Alpha Entity", "score": 0.8}
                ],
            }
        row = {"id": "hotpotqa:q", "question": "Who created Alpha?", "actions": actions}
        first = student_features(row, manifest_names)
        row["actions"][ACTIONS[2]]["policy_features"][2] = 99.0
        repeated = student_features(row, manifest_names)
        self.assertTrue((first == repeated).all())

    def test_single_reader_mode_excludes_answer_history(self) -> None:
        manifest_names = (
            "search_step",
            "retrieval_score",
            "answer_is_unknown",
            "previous_answer_f1",
        )
        actions = {}
        for step, action in enumerate(ACTIONS[:3]):
            actions[action] = {
                "policy_features": [float(step), 0.8, 0.0, 0.2],
                "candidate_evidence": [],
                "evidence_chain": [],
                "prediction": "Alpha",
            }
        row = {"id": "hotpotqa:q", "question": "Who created Alpha?", "actions": actions}
        first = policy_features(row, manifest_names, "single_reader")
        row["actions"][ACTIONS[2]]["policy_features"][3] = 1.0
        repeated = policy_features(row, manifest_names, "single_reader")
        self.assertTrue((first == repeated).all())

    def test_reader_free_cost_calls_only_selected_final_reader(self) -> None:
        actions = {}
        for step, action in enumerate(ACTIONS):
            actions[action] = {
                "reader_latency_seconds": 1.0 + step,
                "reader_generated_tokens": 10 + step,
                "planner_rewrite_latency_cumulative_seconds": 0.5 * step,
            }
        prefix = {
            "initial_retrieval_latency_seconds": 0.25,
            "planner_total_latency_seconds": 1.5,
            "actions": actions,
        }
        reader_free = _costs(prefix, 2, "reader_free_retrieval_student")
        single_reader = _costs(prefix, 3, "single_reader_selective_student")
        self.assertEqual(reader_free["adaptive_latency"], 4.25)
        self.assertEqual(reader_free["adaptive_tokens"], 12.0)
        self.assertEqual(reader_free["reader_calls"], 1.0)
        self.assertEqual(single_reader["adaptive_tokens"], 25.0)
        self.assertEqual(single_reader["reader_calls"], 2.0)


if __name__ == "__main__":
    unittest.main()
