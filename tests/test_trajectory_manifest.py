import json
import tempfile
import unittest
from pathlib import Path

from experiments.evaluate_iterative_hop_retrieval import _manifest_matches


class TrajectoryManifestTest(unittest.TestCase):
    def test_exact_v5_five_source_configuration_is_reusable(self):
        saved = {
            "format_version": 5,
            "experiment": "state_based_stop_continue_retrieval_v5",
            "dataset": "hotpotqa",
            "top_k": 5,
            "query_seed": 42,
            "queries_by_hop": {"2": 1000},
        }
        expected = {
            "format_version": 6,
            "experiment": "structured_gap_evidence_survival_retrieval_v6",
            "dataset": "hotpotqa",
            "top_k": 5,
            "query_seed": 42,
            "queries_by_hop": {"2": 2000},
            "planner_anchor_slots": 2,
            "planner_reader_source_cap": 5,
            "planner_additions_per_step": 1,
            "planner_replace_when_full": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            self.assertTrue(_manifest_matches(path, expected))

    def test_v5_is_not_reused_for_growing_evidence_configuration(self):
        saved = {
            "format_version": 5,
            "experiment": "state_based_stop_continue_retrieval_v5",
            "top_k": 5,
        }
        expected = {
            "format_version": 6,
            "experiment": "structured_gap_evidence_survival_retrieval_v6",
            "top_k": 5,
            "planner_anchor_slots": 2,
            "planner_reader_source_cap": 8,
            "planner_additions_per_step": 1,
            "planner_replace_when_full": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            self.assertFalse(_manifest_matches(path, expected))


if __name__ == "__main__":
    unittest.main()
