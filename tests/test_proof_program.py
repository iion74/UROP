import json
import unittest

from src.proof_program import assemble_bound_program, audit_counterfactuals, execute_program


class ProofProgramTests(unittest.TestCase):
    def test_chain_executes_terminal_object_and_every_dependency_is_necessary(self) -> None:
        context = "[1] Film\nFilm was directed by Person.\n\n[2] Person\nPerson died in Nice.\n\n[3] Noise\nNoise."
        payload = {
            "operation": "chain",
            "steps": [
                {"id": 1, "subject": "Film", "relation": "director", "object": "Person", "evidence_id": 1, "depends_on": []},
                {"id": 2, "subject": "Person", "relation": "place of death", "object": "Nice", "evidence_id": 2, "depends_on": [1]},
            ],
            "answer_step": 2,
            "answer_node": "Nice",
            "comparison": {"operator": "none", "left_step": 0, "left_entity": "", "left_value": "", "right_step": 0, "right_entity": "", "right_value": ""},
        }
        prediction, required, necessity, invariant = audit_counterfactuals(payload, "Where did Film's director die?", context)
        self.assertEqual(prediction, "Nice")
        self.assertEqual(required, (1, 2))
        self.assertEqual(necessity, 1.0)
        self.assertTrue(invariant)

    def test_comparison_is_computed_not_copied_from_answer_node(self) -> None:
        context = "[1] A\nA was released in 1999.\n\n[2] B\nB was released in 2005."
        payload = {
            "operation": "compare",
            "steps": [
                {"id": 1, "subject": "A", "relation": "release year", "object": "1999", "evidence_id": 1, "depends_on": []},
                {"id": 2, "subject": "B", "relation": "release year", "object": "2005", "evidence_id": 2, "depends_on": []},
            ],
            "answer_step": 2,
            "answer_node": "wrong",
            "comparison": {"operator": "later", "left_step": 1, "left_entity": "A", "left_value": "1999", "right_step": 2, "right_entity": "B", "right_value": "2005"},
        }
        prediction, required = execute_program(payload, "Which was released more recently, A or B?", context)
        self.assertEqual(prediction, "B")
        self.assertEqual(required, (1, 2))

    def test_invalid_citation_is_rejected(self) -> None:
        payload = {
            "operation": "lookup",
            "steps": [{"id": 1, "subject": "A", "relation": "is", "object": "B", "evidence_id": 9, "depends_on": []}],
            "answer_step": 1,
            "answer_node": "B",
            "comparison": {"operator": "none", "left_step": 0, "left_entity": "", "left_value": "", "right_step": 0, "right_entity": "", "right_value": ""},
        }
        self.assertEqual(execute_program(payload, "Q", "[1] A\nA is B."), (None, ()))

    def test_binding_cannot_change_question_only_contract_structure(self) -> None:
        contract = {
            "operation": "chain",
            "steps": [
                {"id": 1, "input": "Film", "relation": "director", "output": "$person", "depends_on": []},
                {"id": 2, "input": "$person", "relation": "place of death", "output": "$answer", "depends_on": [1]},
            ],
            "answer_step": 2,
            "comparison": {"operator": "none", "left_step": 0, "left_entity": "", "right_step": 0, "right_entity": ""},
        }
        bindings = {
            "bindings": [
                {"step_id": 1, "subject": "Film", "object": "Person", "evidence_id": 1, "relation": "malicious edit"},
                {"step_id": 2, "subject": "Person", "object": "Nice", "evidence_id": 2},
            ],
            "answer_node": "Nice",
        }
        program = assemble_bound_program(contract, bindings)
        self.assertEqual(program["steps"][0]["relation"], "director")
        self.assertEqual(program["steps"][1]["depends_on"], [1])


if __name__ == "__main__":
    unittest.main()
