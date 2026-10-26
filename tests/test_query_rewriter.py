import tempfile
import unittest
from pathlib import Path

from src.query_rewriter import (
    FrozenOllamaQueryRewriter,
    QueryRewriteCache,
    QueryRewriterConfig,
    clean_search_query,
)


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "response": (
                '{"query":"Where was Nabil Kanso born?","sufficient":false,'
                '"missing_information":"Nabil Kanso birthplace"}'
            ),
            "prompt_eval_count": 10,
            "eval_count": 5,
        }


class _Session:
    def __init__(self):
        self.calls = 0

    def post(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return _Response()


class QueryRewriterTests(unittest.TestCase):
    def test_json_query_is_cleaned(self):
        self.assertEqual(clean_search_query('{"query":"Where is Beirut?"}'), "Where is Beirut?")

    def test_identical_prompt_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            session = _Session()
            with QueryRewriteCache(Path(directory) / "cache.sqlite3") as cache:
                rewriter = FrozenOllamaQueryRewriter(
                    QueryRewriterConfig(host="http://localhost", model="qwen3:8b"),
                    cache,
                    session=session,
                )
                first = rewriter.rewrite("Question", "Evidence")
                second = rewriter.rewrite("Question", "Evidence")
        self.assertEqual(first.query, "Where was Nabil Kanso born?")
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertFalse(first.sufficient)
        self.assertEqual(first.missing_information, "Nabil Kanso birthplace")
        self.assertFalse(second.sufficient)
        self.assertEqual(session.calls, 1)


if __name__ == "__main__":
    unittest.main()
