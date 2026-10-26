import unittest

from experiments.evaluate_reader_routed_policy import _route_records


class ReaderRoutedPolicyTests(unittest.TestCase):
    def test_chain_reader_is_used_only_for_gated_terminal_rows(self) -> None:
        policy = [
            {
                "id": "m:1",
                "dataset": "musique",
                "selected_action": "planned_continue_3",
                "stop_step": 3,
                "policy_em": 0.0,
                "policy_f1": 0.0,
            },
            {
                "id": "m:2",
                "dataset": "musique",
                "selected_action": "planned_continue_2",
                "stop_step": 2,
                "policy_em": 1.0,
                "policy_f1": 1.0,
            },
        ]
        baseline = {
            "m:1": {"actions": {"vector": {"em": 0, "f1": 0}, "planned_continue_3": {"em": 0, "f1": 0}}},
            "m:2": {"actions": {"vector": {"em": 0, "f1": 0}, "planned_continue_2": {"em": 1, "f1": 1}}},
        }
        chain = {"m:1": {"actions": {"planned_continue_3": {"em": 1, "f1": 1}}}}

        rows = _route_records(policy, baseline, chain, ("musique",))

        self.assertEqual(rows[0]["reader_route"], "evidence_chain_v3")
        self.assertEqual(rows[0]["policy_f1"], 1.0)
        self.assertEqual(rows[1]["reader_route"], "baseline_v2")
        self.assertEqual(rows[1]["policy_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
