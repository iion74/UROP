import unittest

from src.assistant import build_answer_prompt


class AssistantPromptTests(unittest.TestCase):
    def test_answer_prompt_requires_evidence_chain_and_no_guessing(self) -> None:
        prompt = build_answer_prompt("질문", "[근거 1] 내용")

        self.assertIn("가장 짧은 근거 사슬", prompt)
        self.assertIn("동명이인", prompt)
        self.assertIn("추측", prompt)
        self.assertIn("[근거 1] 내용", prompt)


if __name__ == "__main__":
    unittest.main()
