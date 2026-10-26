from types import SimpleNamespace

import numpy as np

from experiments.evaluate_sequential_answer_gain_policy import _features_by_step
from experiments.train_adaptive_proof_completion_policy import ACTIONS
from src.answer_gain_policy import (
    CONTEXT_PREFIX_FEATURE_NAMES,
    AnswerGainPolicy,
    StepwiseSearcher,
    sequential_state,
)
from src.document_loader import Chunk


def _actions():
    actions = []
    for step in range(len(ACTIONS)):
        values = [0.1 * step] * len(CONTEXT_PREFIX_FEATURE_NAMES)
        actions.append({
            "policy_features": values,
            "prediction": f"answer{step}",
            "evidence_chain": [f"Alpha was born in {1990 + step}"],
            "candidate_evidence": [{"content_id": f"doc{step}", "title": f"Alpha {step}", "score": 0.5}],
        })
    return actions


def test_live_state_matches_training_state_for_every_prefix():
    actions = _actions()
    question = "Was Alpha born earlier or later than Beta?"
    row = {"id": "q", "question": question, "actions": dict(zip(ACTIONS, actions, strict=True))}
    training = _features_by_step(row, CONTEXT_PREFIX_FEATURE_NAMES)
    for step in range(3):
        np.testing.assert_array_equal(sequential_state(question, actions[: step + 1], step), training[step])


class _Retriever:
    def __init__(self):
        self.chunks = [Chunk(f"c{index}", f"c{index}", f"Alpha fact {index}", f"Alpha {index}")
                       for index in range(30)]

    def search(self, query, top_k=5):
        offset = sum(map(ord, query)) % 10
        return [(chunk, 1.0 - index / 40) for index, chunk in enumerate(self.chunks[offset:offset + top_k])]


class _Rewriter:
    def rewrite(self, question, evidence, previous_queries=()):
        return SimpleNamespace(query=f"{question} hop {len(previous_queries)}", sufficient=None,
                               missing_information="", latency_seconds=0.0)


class _Reader:
    def __init__(self):
        self.calls = 0

    def answer(self, question, context):
        self.calls += 1
        return SimpleNamespace(raw_response='{"evidence_chain":["Alpha is a fact"],"answer":"Alpha"}',
                               prediction="Alpha")


class _GainModel:
    def __init__(self, gains):
        self.gains = list(gains)

    def predict(self, states):
        assert states.shape == (1, 105)
        return np.asarray([self.gains.pop(0)])


def _run(gains):
    reader = _Reader()
    policy = AnswerGainPolicy(_GainModel(gains), (0.03, 0.01, 0.0), "test")
    return StepwiseSearcher(_Retriever(), _Rewriter(), reader, policy).run("Who is Alpha?"), reader


def test_policy_stops_when_predicted_gain_is_not_above_threshold():
    result, reader = _run([0.05, 0.005])
    assert [step.decision for step in result.steps] == ["SEARCH", "STOP"]
    assert result.total_searches == 2 and reader.calls == 2
    assert result.steps[1].query == "Who is Alpha? hop 0"


def test_policy_searches_at_most_four_times():
    result, reader = _run([1.0, 1.0, 1.0])
    assert [step.decision for step in result.steps] == ["SEARCH", "SEARCH", "SEARCH", "LIMIT"]
    assert result.total_searches == 4 and reader.calls == 4
    assert result.final.predicted_gain is None
