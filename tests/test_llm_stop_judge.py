import json
import tempfile
import unittest
from pathlib import Path

from src.llm_stop_judge import (
    FrozenOllamaStopJudge,
    StopJudgeCache,
    StopJudgeConfig,
    parse_stop_judgment,
    stop_judge_prompt,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append((url, json, timeout))
        return _Response(self.payload)


class LlmStopJudgeTests(unittest.TestCase):
    def test_parse_valid_stop(self):
        raw = json.dumps(
            {
                "decision": "STOP",
                "supported_answer": "Paris",
                "missing_information": "",
                "confidence": 0.9,
            }
        )
        decision, answer, missing, confidence, valid = parse_stop_judgment(raw)
        self.assertEqual((decision, answer, missing), ("STOP", "Paris", ""))
        self.assertAlmostEqual(confidence, 0.9)
        self.assertTrue(valid)

    def test_malformed_and_empty_stop_continue_conservatively(self):
        self.assertEqual(parse_stop_judgment("not json")[0], "CONTINUE")
        decision, _, _, _, valid = parse_stop_judgment(
            '{"decision":"STOP","supported_answer":"","missing_information":"","confidence":1}'
        )
        self.assertEqual(decision, "CONTINUE")
        self.assertFalse(valid)

    def test_prompt_contains_runtime_state_but_no_gold_fields(self):
        prompt = stop_judge_prompt("Who won?", "[1] Source\nEvidence", ("first query",))
        self.assertIn("Who won?", prompt)
        self.assertIn("Evidence", prompt)
        self.assertIn("first query", prompt)
        self.assertNotIn("gold answer", prompt.casefold())
        self.assertNotIn("hop count", prompt.casefold())

    def test_ollama_call_is_deterministic_and_cached(self):
        payload = {
            "response": json.dumps(
                {
                    "decision": "CONTINUE",
                    "supported_answer": "",
                    "missing_information": "winner",
                    "confidence": 0.8,
                }
            ),
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        session = _Session(payload)
        with tempfile.TemporaryDirectory() as directory:
            with StopJudgeCache(Path(directory) / "cache.sqlite3") as cache:
                judge = FrozenOllamaStopJudge(
                    StopJudgeConfig(host="http://test", model="qwen3:8b"),
                    cache,
                    session=session,
                )
                first = judge.judge("question", "evidence")
                second = judge.judge("question", "evidence")
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(len(session.calls), 1)
        request = session.calls[0][1]
        self.assertFalse(request["think"])
        self.assertEqual(request["options"]["temperature"], 0.0)
        self.assertEqual(request["options"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
