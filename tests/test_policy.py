import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.policy import LearnedActionPolicy, LinearActionPolicy, LinearUtilityPolicy, POLICY_FEATURE_NAMES, profile_features
from src.routing import EvidenceProfile, QueryRouter


class LearnedActionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vector_profile = EvidenceProfile(0.95, 0.90, 0.05, 0.05, 0.80, 0.05, 0.05)
        self.graph_profile = EvidenceProfile(0.05, 0.05, 0.90, 0.90, 0.20, 0.60, 0.95)

    def test_fit_save_load_and_decide(self) -> None:
        features = np.vstack(
            [
                *[profile_features(self.vector_profile) for _ in range(20)],
                *[profile_features(self.graph_profile) for _ in range(20)],
            ]
        )
        labels = ["vector"] * 20 + ["graph_2hop_large"] * 20
        model = LinearActionPolicy.fit(features, labels, epochs=120, learning_rate=0.1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            model.save(path)
            loaded = LinearActionPolicy.load(path)

        action, confidence = loaded.predict(profile_features(self.graph_profile))
        self.assertEqual(action, "graph_2hop_large")
        self.assertGreater(confidence, 0.5)

        learned = LearnedActionPolicy(loaded, QueryRouter(), min_confidence=0.0)
        decision = learned.decide(self.graph_profile)
        self.assertEqual(decision.route, "graph_2hop")
        self.assertIsNotNone(decision.policy_confidence)

    def test_feature_shape_is_stable(self) -> None:
        features = profile_features(self.vector_profile)
        self.assertEqual(features.shape, (len(POLICY_FEATURE_NAMES),))

    def test_utility_policy_learns_rare_graph_advantage(self) -> None:
        features = np.vstack(
            [
                *[profile_features(self.vector_profile) for _ in range(20)],
                *[profile_features(self.graph_profile) for _ in range(8)],
            ]
        )
        utilities = np.tile(np.asarray([0.9, 0.7, 0.6, 0.5, 0.4]), (len(features), 1))
        utilities[20:, :] = np.asarray([0.4, 0.5, 0.7, 0.8, 0.95])
        weights = np.asarray([1.0] * 20 + [8.0] * 8)
        model = LinearUtilityPolicy.fit(features, utilities, sample_weights=weights, epochs=160, learning_rate=0.1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utility_policy.npz"
            model.save(path)
            loaded = LinearUtilityPolicy.load(path)

        action, advantage, _ = loaded.choose(profile_features(self.graph_profile))
        self.assertEqual(action, "graph_2hop_large")
        self.assertGreater(advantage, 0.1)

        learned = LearnedActionPolicy(loaded, QueryRouter(), min_expected_advantage=0.0)
        self.assertEqual(learned.decide(self.graph_profile).route, "graph_2hop")


if __name__ == "__main__":
    unittest.main()
