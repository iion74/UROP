"""Answer-aware gate components for learned retrieval policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class QuestionEvidenceGate:
    """Binary Graph-benefit gate over question text and evidence statistics."""

    vectorizer: TfidfVectorizer
    numeric_mean: np.ndarray
    numeric_scale: np.ndarray
    classifier: LogisticRegression

    @classmethod
    def fit(
        cls,
        numeric_features: np.ndarray,
        questions: list[str],
        labels: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        seed: int = 42,
        max_text_features: int = 15_000,
    ) -> "QuestionEvidenceGate":
        numeric_features = np.asarray(numeric_features, dtype="float64")
        labels = np.asarray(labels, dtype="int64")
        if numeric_features.ndim != 2 or len(numeric_features) != len(questions):
            raise ValueError("gate의 numeric feature와 질문 수가 일치해야 합니다.")
        if labels.shape != (len(questions),) or set(np.unique(labels)) != {0, 1}:
            raise ValueError("gate 학습에는 양성과 음성이 모두 필요합니다.")
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype="float64")
            if sample_weights.shape != labels.shape:
                raise ValueError("gate sample weight의 길이가 맞지 않습니다.")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_text_features,
            sublinear_tf=True,
        )
        text_features = vectorizer.fit_transform(questions)
        mean = numeric_features.mean(axis=0)
        scale = numeric_features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        combined = hstack(
            [text_features, csr_matrix((numeric_features - mean) / scale)],
            format="csr",
        )
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
        )
        classifier.fit(combined, labels, sample_weight=sample_weights)
        return cls(vectorizer, mean, scale, classifier)

    def predict_graph_probability(
        self,
        numeric_features: np.ndarray,
        questions: list[str],
    ) -> np.ndarray:
        numeric_features = np.asarray(numeric_features, dtype="float64")
        if numeric_features.ndim != 2 or len(numeric_features) != len(questions):
            raise ValueError("gate 추론 feature와 질문 수가 일치해야 합니다.")
        if numeric_features.shape[1] != len(self.numeric_mean):
            raise ValueError("gate numeric feature 차원이 학습 시점과 다릅니다.")
        text_features = self.vectorizer.transform(questions)
        combined = hstack(
            [
                text_features,
                csr_matrix((numeric_features - self.numeric_mean) / self.numeric_scale),
            ],
            format="csr",
        )
        return self.classifier.predict_proba(combined)[:, 1]
