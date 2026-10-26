import copy

import numpy as np

from experiments.evaluate_context_reader_aware_prefixes import CONTEXT_PREFIX_FEATURE_NAMES
from experiments.evaluate_sequential_answer_gain_policy import (
    _features_by_step,
    _future_gain,
    _simulate,
)
from experiments.train_adaptive_proof_completion_policy import ACTIONS


def _row():
    actions = {}
    for step, name in enumerate(ACTIONS):
        values = [0.0] * len(CONTEXT_PREFIX_FEATURE_NAMES)
        values[0] = float(step)
        values[2] = 0.6 + step / 10
        actions[name] = {
            "policy_features": values,
            "prediction": f"answer{step}",
            "evidence_chain": [f"fact{step}"],
            "candidate_evidence": [
                {"content_id": f"doc{step}", "title": f"Document {step}", "score": 0.6}
            ],
        }
    return {"id": "question1", "question": "Which document supports the answer?", "actions": actions}


def test_early_features_do_not_use_future_searches():
    row = _row()
    original = _features_by_step(row, CONTEXT_PREFIX_FEATURE_NAMES)
    changed = copy.deepcopy(row)
    changed["actions"][ACTIONS[2]]["policy_features"][2] = 0.01
    changed["actions"][ACTIONS[2]]["prediction"] = "changed"
    changed["actions"][ACTIONS[3]]["prediction"] = "also changed"
    revised = _features_by_step(changed, CONTEXT_PREFIX_FEATURE_NAMES)
    assert original.shape == (3, 105)
    np.testing.assert_array_equal(original[:2], revised[:2])
    assert not np.array_equal(original[2], revised[2])


def test_future_gain_and_sequential_stop():
    scores = np.asarray([[0.5, 0.4, 0.8, 0.6], [0.9, 0.3, 0.2, 0.1]])
    np.testing.assert_allclose(_future_gain(scores), [[0.3, 0.4, -0.2], [-0.6, -0.1, -0.1]])
    predictions = np.asarray([[0.2, 0.1, -0.1], [-0.2, 0.1, 0.1]])
    np.testing.assert_array_equal(_simulate(predictions, np.zeros(3)), [2, 0])
