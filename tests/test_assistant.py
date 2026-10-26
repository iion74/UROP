import json
import unittest

from src.assistant import build_answer_prompt, parse_quiz


class AssistantPromptTests(unittest.TestCase):
    def test_quiz_keeps_only_gradable_four_option_items(self) -> None:
        good = {"question": "Q1", "options": ["a", "b", "c", "d"], "answer_index": 2,
                "explanation": "e", "evidence": 1}
        raw = json.dumps({"questions": [good, {**good, "options": ["a", "b"]}, {**good, "answer_index": 7}]})
        quiz = parse_quiz(raw, "topic", "unit")
        self.assertEqual(len(quiz.items), 1)
        self.assertEqual(quiz.items[0].options[quiz.items[0].answer_index], "c")
        self.assertIn("해석하지 못했습니다", parse_quiz("not json", "topic", None).message)

    def test_answer_prompt_requires_evidence_chain_and_no_guessing(self) -> None:
        prompt = build_answer_prompt("질문", "[근거 1] 내용")

        self.assertIn("가장 짧은 근거 사슬", prompt)
        self.assertIn("동명이인", prompt)
        self.assertIn("추측", prompt)
        self.assertIn("[근거 1] 내용", prompt)


if __name__ == "__main__":
    unittest.main()
