"""Small, reproducible action policies for adaptive retrieval.

The policy is intentionally a NumPy softmax classifier rather than another
language model.  It learns which bounded retrieval action recovers the most
gold evidence for a query, while its runtime decision remains cheap enough for
the local MVP.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from .routing import POLICY_ACTION_IDS, EvidenceProfile, QueryRouter, RouteDecision


POLICY_FEATURE_NAMES = (
    "score_concentration",
    "score_margin",
    "source_dispersion",
    "graph_connectivity",
    "redundancy",
    "one_hop_gain",
    "two_hop_gain",
    "uncertainty",
)


@dataclass(frozen=True)
class PolicyTrainingProgress:
    """A terminal/UI-neutral snapshot emitted while fitting the policy."""

    epoch: int
    epochs: int
    loss: float
    accuracy: float
    elapsed_seconds: float
    eta_seconds: float
    selected_utility: float | None = None


def profile_features(profile: EvidenceProfile) -> np.ndarray:
    """Create the exact feature vector shared by training and MVP inference."""
    return np.asarray(
        (
            profile.score_concentration,
            profile.score_margin,
            profile.source_dispersion,
            profile.graph_connectivity,
            profile.redundancy,
            profile.one_hop_gain,
            profile.two_hop_gain,
            profile.uncertainty,
        ),
        dtype="float64",
    )


class LinearActionPolicy:
    """A dependency-free multiclass softmax policy trained on retrieval labels."""

    def __init__(
        self,
        actions: tuple[str, ...],
        mean: np.ndarray,
        scale: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        self.actions = actions
        self.mean = mean
        self.scale = scale
        self.weights = weights
        self.bias = bias

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: Iterable[str],
        *,
        epochs: int = 80,
        learning_rate: float = 0.08,
        l2: float = 1e-4,
        batch_size: int = 256,
        seed: int = 42,
        progress_callback: Callable[[PolicyTrainingProgress], None] | None = None,
        report_every: int = 5,
    ) -> "LinearActionPolicy":
        features = np.asarray(features, dtype="float64")
        labels = tuple(labels)
        if features.ndim != 2 or features.shape[1] != len(POLICY_FEATURE_NAMES):
            raise ValueError(f"정책 입력은 ({len(POLICY_FEATURE_NAMES)})개 특징을 가져야 합니다.")
        if len(features) == 0 or len(features) != len(labels):
            raise ValueError("비어 있지 않은 동일 길이의 특징과 라벨이 필요합니다.")
        unknown = set(labels) - set(POLICY_ACTION_IDS)
        if unknown:
            raise ValueError(f"지원하지 않는 행동 라벨입니다: {sorted(unknown)}")

        action_to_index = {action: index for index, action in enumerate(POLICY_ACTION_IDS)}
        targets = np.asarray([action_to_index[label] for label in labels], dtype="int64")
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalised = (features - mean) / scale
        rng = np.random.default_rng(seed)
        weights = rng.normal(0.0, 0.01, size=(normalised.shape[1], len(POLICY_ACTION_IDS)))
        bias = np.zeros(len(POLICY_ACTION_IDS), dtype="float64")
        if report_every < 1:
            raise ValueError("report_every는 1 이상이어야 합니다.")

        started = perf_counter()
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(normalised))
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                batch_features = normalised[batch]
                probabilities = _softmax(batch_features @ weights + bias)
                probabilities[np.arange(len(batch)), targets[batch]] -= 1.0
                gradient_weights = batch_features.T @ probabilities / len(batch) + l2 * weights
                gradient_bias = probabilities.mean(axis=0)
                weights -= learning_rate * gradient_weights
                bias -= learning_rate * gradient_bias
            if progress_callback and (epoch == 1 or epoch % report_every == 0 or epoch == epochs):
                probabilities = _softmax(normalised @ weights + bias)
                loss = float(-np.log(probabilities[np.arange(len(targets)), targets] + 1e-12).mean())
                accuracy = float((np.argmax(probabilities, axis=1) == targets).mean())
                elapsed_seconds = perf_counter() - started
                eta_seconds = elapsed_seconds / epoch * (epochs - epoch)
                progress_callback(
                    PolicyTrainingProgress(
                        epoch=epoch,
                        epochs=epochs,
                        loss=loss,
                        accuracy=accuracy,
                        elapsed_seconds=elapsed_seconds,
                        eta_seconds=eta_seconds,
                    )
                )
        return cls(POLICY_ACTION_IDS, mean, scale, weights, bias)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype="float64")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != len(POLICY_FEATURE_NAMES):
            raise ValueError("저장된 정책과 입력 특징의 차원이 다릅니다.")
        return _softmax(((features - self.mean) / self.scale) @ self.weights + self.bias)

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        probabilities = self.predict_proba(features)[0]
        index = int(np.argmax(probabilities))
        return self.actions[index], float(probabilities[index])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            actions=np.asarray(self.actions),
            feature_names=np.asarray(POLICY_FEATURE_NAMES),
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
            bias=self.bias,
        )

    @classmethod
    def load(cls, path: Path) -> "LinearActionPolicy":
        with np.load(path, allow_pickle=False) as payload:
            actions = tuple(str(item) for item in payload["actions"].tolist())
            feature_names = tuple(str(item) for item in payload["feature_names"].tolist())
            if actions != POLICY_ACTION_IDS:
                raise ValueError("정책 모델의 행동 공간이 현재 시스템과 다릅니다.")
            if feature_names != POLICY_FEATURE_NAMES:
                raise ValueError("정책 모델의 특징 정의가 현재 시스템과 다릅니다.")
            return cls(actions, payload["mean"], payload["scale"], payload["weights"], payload["bias"])


class LinearUtilityPolicy:
    """Predict the utility of every retrieval action instead of one class.

    A multi-hop graph action is useful only for a minority of questions.  A
    plain best-action classifier can collapse to ``vector`` under this natural
    imbalance.  This model receives the complete oracle utility vector during
    training and chooses the action with the largest *predicted* utility at
    inference time.
    """

    MODEL_TYPE = "utility_regression_v1"

    def __init__(
        self,
        actions: tuple[str, ...],
        mean: np.ndarray,
        scale: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        self.actions = actions
        self.mean = mean
        self.scale = scale
        self.weights = weights
        self.bias = bias

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        utilities: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        epochs: int = 80,
        learning_rate: float = 0.08,
        l2: float = 1e-4,
        batch_size: int = 256,
        seed: int = 42,
        progress_callback: Callable[[PolicyTrainingProgress], None] | None = None,
        report_every: int = 5,
    ) -> "LinearUtilityPolicy":
        features = np.asarray(features, dtype="float64")
        utilities = np.asarray(utilities, dtype="float64")
        if features.ndim != 2 or features.shape[1] != len(POLICY_FEATURE_NAMES):
            raise ValueError(f"정책 입력은 ({len(POLICY_FEATURE_NAMES)})개 특징을 가져야 합니다.")
        if utilities.shape != (len(features), len(POLICY_ACTION_IDS)):
            raise ValueError("행동 효용은 (질문 수, 행동 수) 형태여야 합니다.")
        if len(features) == 0:
            raise ValueError("비어 있지 않은 학습 데이터가 필요합니다.")
        if report_every < 1:
            raise ValueError("report_every는 1 이상이어야 합니다.")

        if sample_weights is None:
            sample_weights = np.ones(len(features), dtype="float64")
        else:
            sample_weights = np.asarray(sample_weights, dtype="float64")
            if sample_weights.shape != (len(features),) or np.any(sample_weights <= 0):
                raise ValueError("sample_weights는 질문 수와 같은 길이의 양수 벡터여야 합니다.")

        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalised = (features - mean) / scale
        rng = np.random.default_rng(seed)
        weights = rng.normal(0.0, 0.01, size=(normalised.shape[1], len(POLICY_ACTION_IDS)))
        bias = np.average(utilities, axis=0, weights=sample_weights)
        started = perf_counter()

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(normalised))
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                batch_features = normalised[batch]
                batch_weights = sample_weights[batch]
                errors = batch_features @ weights + bias - utilities[batch]
                normaliser = float(batch_weights.sum())
                weighted_errors = errors * batch_weights[:, None]
                gradient_weights = batch_features.T @ weighted_errors / normaliser + l2 * weights
                gradient_bias = weighted_errors.sum(axis=0) / normaliser
                weights -= learning_rate * gradient_weights
                bias -= learning_rate * gradient_bias

            if progress_callback and (epoch == 1 or epoch % report_every == 0 or epoch == epochs):
                predicted_utilities = normalised @ weights + bias
                selected = np.argmax(predicted_utilities, axis=1)
                oracle = np.argmax(utilities, axis=1)
                per_example_loss = np.mean((predicted_utilities - utilities) ** 2, axis=1)
                loss = float(np.average(per_example_loss, weights=sample_weights))
                selected_utility = float(np.mean(utilities[np.arange(len(utilities)), selected]))
                elapsed_seconds = perf_counter() - started
                eta_seconds = elapsed_seconds / epoch * (epochs - epoch)
                progress_callback(
                    PolicyTrainingProgress(
                        epoch=epoch,
                        epochs=epochs,
                        loss=loss,
                        accuracy=float((selected == oracle).mean()),
                        elapsed_seconds=elapsed_seconds,
                        eta_seconds=eta_seconds,
                        selected_utility=selected_utility,
                    )
                )
        return cls(POLICY_ACTION_IDS, mean, scale, weights, bias)

    def predict_utilities(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype="float64")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != len(POLICY_FEATURE_NAMES):
            raise ValueError("저장된 정책과 입력 특징의 차원이 다릅니다.")
        return ((features - self.mean) / self.scale) @ self.weights + self.bias

    def choose(self, features: np.ndarray) -> tuple[str, float, np.ndarray]:
        utilities = self.predict_utilities(features)[0]
        selected_index = int(np.argmax(utilities))
        vector_index = self.actions.index("vector")
        advantage = float(utilities[selected_index] - utilities[vector_index])
        return self.actions[selected_index], advantage, utilities

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            model_type=np.asarray(self.MODEL_TYPE),
            actions=np.asarray(self.actions),
            feature_names=np.asarray(POLICY_FEATURE_NAMES),
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
            bias=self.bias,
        )

    @classmethod
    def load(cls, path: Path) -> "LinearUtilityPolicy":
        with np.load(path, allow_pickle=False) as payload:
            model_type = str(payload["model_type"].item())
            actions = tuple(str(item) for item in payload["actions"].tolist())
            feature_names = tuple(str(item) for item in payload["feature_names"].tolist())
            if model_type != cls.MODEL_TYPE:
                raise ValueError("저장된 모델은 효용 정책 형식이 아닙니다.")
            if actions != POLICY_ACTION_IDS or feature_names != POLICY_FEATURE_NAMES:
                raise ValueError("정책 모델의 행동 또는 특징 정의가 현재 시스템과 다릅니다.")
            return cls(actions, payload["mean"], payload["scale"], payload["weights"], payload["bias"])


class LearnedActionPolicy:
    """Run a saved policy while preserving the transparent rule fallback."""

    def __init__(
        self,
        model: LinearActionPolicy | LinearUtilityPolicy,
        fallback: QueryRouter,
        min_confidence: float = 0.45,
        min_expected_advantage: float = 0.01,
    ) -> None:
        self.model = model
        self.fallback = fallback
        self.min_confidence = min_confidence
        self.min_expected_advantage = min_expected_advantage

    @classmethod
    def from_path(
        cls,
        path: Path,
        fallback: QueryRouter,
        min_confidence: float = 0.45,
        min_expected_advantage: float = 0.01,
    ) -> "LearnedActionPolicy":
        with np.load(path, allow_pickle=False) as payload:
            model_type = str(payload["model_type"].item()) if "model_type" in payload.files else "classifier_v1"
        model = LinearUtilityPolicy.load(path) if model_type == LinearUtilityPolicy.MODEL_TYPE else LinearActionPolicy.load(path)
        return cls(model, fallback, min_confidence, min_expected_advantage)

    def decide(self, profile: EvidenceProfile) -> RouteDecision:
        if isinstance(self.model, LinearUtilityPolicy):
            action_id, advantage, utilities = self.model.choose(profile_features(profile))
            if action_id != "vector" and advantage < self.min_expected_advantage:
                action_id = "vector"
                reason = "learned_utility_gain_below_threshold"
            else:
                reason = "learned_utility_policy"
            selected_utility = float(utilities[self.model.actions.index(action_id)])
            decision = self.fallback.decision_for_action(profile, action_id, reason=reason)
            return replace(
                decision,
                expected_utility=selected_utility,
                reasons=(*decision.reasons, f"expected_advantage={advantage:.4f}"),
            )

        action_id, confidence = self.model.predict(profile_features(profile))
        if confidence < self.min_confidence:
            fallback = self.fallback.decide(profile)
            return replace(
                fallback,
                reasons=("learned_policy_low_confidence", action_id, *fallback.reasons),
                policy_confidence=confidence,
            )
        return self.fallback.decision_for_action(
            profile,
            action_id,
            policy_confidence=confidence,
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
