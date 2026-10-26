import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.answer_verifier import (
    FrozenGroundedAnswerVerifier,
    _parse_adjudication,
    _parse_response,
    adjudicator_prompt,
    verifier_prompt,
)
from src.qa_evaluation import AnswerCache, ReaderConfig


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "response": '{"verification_chain":["A supports B"],"decision":"correct","answer":"B"}',
            "prompt_eval_count": 10,
            "eval_count": 5,
        }


class FakeSession:
    def __init__(self):
        self.calls = 0

    def post(self, url, json, timeout):
        del url, json, timeout
        self.calls += 1
        return FakeResponse()


class AnswerVerifierTests(unittest.TestCase):
    def test_keep_decision_always_preserves_exact_draft(self) -> None:
        raw = '{"verification_chain":["fact"],"decision":"keep","answer":"changed"}'
        decision, answer, chain = _parse_response(raw, "Original")

        self.assertEqual((decision, answer), ("keep", "Original"))
        self.assertEqual(chain, ("fact",))

    def test_malformed_response_conservatively_keeps_draft(self) -> None:
        self.assertEqual(_parse_response("not json", "Original")[:2], ("keep", "Original"))

    def test_prompt_requires_one_uniform_conservative_procedure(self) -> None:
        prompt = verifier_prompt("Question", "Evidence", "Draft")
        self.assertIn("regardless of dataset", prompt)
        self.assertIn("Correct the draft only", prompt)

    def test_verifier_caches_identical_request(self) -> None:
        session = FakeSession()
        with TemporaryDirectory() as directory:
            with AnswerCache(Path(directory) / "cache.sqlite3") as cache:
                verifier = FrozenGroundedAnswerVerifier(
                    ReaderConfig(host="http://localhost", model="qwen3:8b"),
                    cache,
                    session=session,
                )
                first = verifier.verify("Q", "E", "A")
                second = verifier.verify("Q", "E", "A")

        self.assertEqual(first.prediction, "B")
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(session.calls, 1)

    def test_adjudication_defaults_to_empty_answer(self) -> None:
        self.assertEqual(_parse_adjudication("not json"), ("", ()))
        answer, chain = _parse_adjudication(
            '{"independent_chain":["fact"],"answer":"Result"}'
        )
        self.assertEqual(answer, "Result")
        self.assertEqual(chain, ("fact",))

    def test_adjudicator_hides_candidates_and_requires_independent_comparison(self) -> None:
        prompt = adjudicator_prompt("Q", "E")
        self.assertIn("no earlier draft", prompt)
        self.assertIn("both compared values", prompt)


if __name__ == "__main__":
    unittest.main()
