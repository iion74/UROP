import unittest

from experiments.evaluate_iterative_hop_retrieval import _planned_stop_labels


class StopContinueLabelTests(unittest.TestCase):
    def test_labels_compare_current_prefix_with_all_future_prefixes(self):
        results = {
            "vector": {"all_support_recall": 0.0, "source_recall": 0.25},
            "planned_continue_1": {"all_support_recall": 0.0, "source_recall": 0.50},
            "planned_continue_2": {"all_support_recall": 1.0, "source_recall": 1.00},
            "planned_continue_3": {"all_support_recall": 0.0, "source_recall": 0.75},
        }

        labels = _planned_stop_labels(results, 3)

        self.assertEqual([row["label"] for row in labels], ["CONTINUE", "CONTINUE", "STOP"])
        self.assertEqual(labels[0]["best_future_action"], "planned_continue_2")

    def test_tie_prefers_stop(self):
        result = {"all_support_recall": 1.0, "source_recall": 1.0}
        results = {
            "vector": result,
            "planned_continue_1": result,
            "planned_continue_2": result,
        }

        labels = _planned_stop_labels(results, 2)

        self.assertEqual([row["label"] for row in labels], ["STOP", "STOP"])


if __name__ == "__main__":
    unittest.main()
