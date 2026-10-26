import unittest
from types import SimpleNamespace

from experiments.evaluate_llm_stop_policy import ACTIONS, _evaluate_row
from src.document_loader import Chunk
from src.llm_stop_judge import StopJudgeResult


class _Judge:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def judge(self, question, evidence, previous_queries=()):
        self.calls.append((question, evidence, previous_queries))
        decision = next(self.decisions)
        return StopJudgeResult(
            decision=decision,
            supported_answer="answer" if decision == "STOP" else "",
            missing_information="missing" if decision == "CONTINUE" else "",
            confidence=0.9,
            raw_response="{}",
            valid=True,
            cache_hit=False,
            latency_seconds=0.1,
            prompt_eval_count=10,
            eval_count=5,
        )


def _fixture():
    chunks = {}
    actions = {}
    answer_actions = {}
    for step, action in enumerate(ACTIONS):
        chunk_id = f"chunk-{step}"
        chunks[chunk_id] = Chunk(chunk_id=chunk_id, source=f"source-{step}", title=f"Title {step}", text="text")
        actions[action] = {
            "evidence": [
                {"chunk_id": chunk_id, "content_id": f"source-{step}", "title": f"Title {step}", "score": 0.5}
            ]
        }
        answer_actions[action] = {
            "em": step / 3,
            "f1": step / 3,
            "source_recall": 1.0,
            "all_support_recall": 1.0,
            "context_characters": 10,
            "reader_latency_seconds": 0.2,
        }
    row = {
        "id": "id",
        "dataset": "hotpotqa",
        "question": "question",
        "direct_em": 0.0,
        "direct_f1": 0.0,
        "oracle_f1": 1.0,
        "actions": answer_actions,
        "planner_audit": [
            {"query": f"query-{step}", "latency_seconds": 0.3}
            for step in range(3)
        ],
    }
    trajectory = {"actions": actions, "planner_audit": row["planner_audit"]}
    by_source = {chunk.source: [chunk] for chunk in chunks.values()}
    args = SimpleNamespace(
        context_character_budget=6000,
        context_allocation="sequential",
        context_source_order="trajectory",
        context_compression="none",
        focused_sentences_per_source=2,
    )
    return row, trajectory, chunks, by_source, args


class EvaluateLlmStopPolicyTests(unittest.TestCase):
    def test_stops_at_first_supported_state(self):
        row, trajectory, chunks, by_source, args = _fixture()
        judge = _Judge(["CONTINUE", "STOP"])
        result = _evaluate_row(judge, row, trajectory, chunks, by_source, args)
        self.assertEqual(result["stop_step"], 1)
        self.assertEqual(result["selected_action"], ACTIONS[1])
        self.assertEqual(result["judge_calls"], 2)
        self.assertEqual(judge.calls[1][2], ("query-0",))

    def test_all_continue_reaches_cap_without_fourth_judge_call(self):
        row, trajectory, chunks, by_source, args = _fixture()
        judge = _Judge(["CONTINUE", "CONTINUE", "CONTINUE"])
        result = _evaluate_row(judge, row, trajectory, chunks, by_source, args)
        self.assertEqual(result["stop_step"], 3)
        self.assertEqual(result["judge_calls"], 3)
        self.assertEqual(result["policy_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
