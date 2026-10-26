"""Live STOP/SEARCH control with the paper's answer-gain policy.

The paper policy (``sequential_best_future_f1_gain_hgb``) was trained on cached
trajectories.  This module runs the same decision loop online: mandatory
initial retrieval, a frozen evidence-chain reader answer, the 105-dimensional
observable state, and a stage-specific threshold on the predicted future
answer-F1 gain.  Feature code lives here so training, replay and the web app
share one implementation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from .document_loader import Chunk
from .planned_retrieval import run_planned_retrieval
from .qa_evaluation import FrozenOllamaReader, evidence_context
from .query_rewriter import FrozenOllamaQueryRewriter
from .reader_sufficiency import (
    READER_SUFFICIENCY_FEATURE_NAMES,
    chain_structure_features,
    parse_evidence_chain,
    reader_sufficiency_features,
)


CONTEXT_PREFIX_FEATURE_NAMES = (
    "search_step",
    "candidate_count",
    "retrieval_score_max",
    "retrieval_score_mean",
    "retrieval_score_min",
    "retrieval_score_gap_1_2",
    "new_candidate_fraction",
    "previous_candidate_survival_fraction",
    "planner_sufficient_known",
    "planner_reported_sufficient",
    "planner_missing_information_length",
    "planner_consecutive_no_new",
    *READER_SUFFICIENCY_FEATURE_NAMES,
)
READER_HISTORY_FEATURES = frozenset(
    {
        "previous_answer_exact",
        "previous_answer_f1",
        "initial_answer_exact",
        "initial_answer_f1",
        "answer_history_agreement",
        "previous_chain_jaccard",
    }
)
# Total searches including the mandatory first one; decisions follow 1, 2, 3.
MAX_TOTAL_SEARCHES = 4
DECISION_POINTS = MAX_TOTAL_SEARCHES - 1

_TOKENS = re.compile(r"[a-z0-9]+")
_COMPARISON_TERMS = frozenset(
    {"earlier", "later", "older", "younger", "longer", "more", "less", "same", "both", "first"}
)
_BOOLEAN_STARTS = frozenset({"am", "are", "can", "did", "do", "does", "has", "have", "is", "was", "were"})
_WH_WORDS = ("who", "when", "where", "which", "what", "why", "how")


def words(text: str) -> set[str]:
    return set(_TOKENS.findall(str(text).lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def retrieval_features(
    search_step: int,
    sources,
    previous_ids: set[str],
    planner_step,
) -> np.ndarray:
    scores = sorted((float(score) for _, score in sources), reverse=True)
    current_ids = {chunk.source for chunk, _ in sources}
    sufficient = planner_step.rewrite.sufficient if planner_step is not None else None
    missing = planner_step.rewrite.missing_information if planner_step is not None else ""
    consecutive_no_new = (
        planner_step.state.consecutive_no_new + int(not planner_step.added_chunks)
        if planner_step is not None
        else 0
    )
    return np.asarray(
        [
            float(search_step),
            float(len(sources)),
            max(scores, default=0.0),
            float(np.mean(scores)) if scores else 0.0,
            min(scores, default=0.0),
            scores[0] - scores[1] if len(scores) > 1 else max(scores, default=0.0),
            len(current_ids - previous_ids) / max(1, len(current_ids)),
            len(current_ids & previous_ids) / max(1, len(previous_ids)),
            float(sufficient is not None),
            float(sufficient is True),
            min(1.0, len(str(missing).split()) / 24.0),
            min(1.0, consecutive_no_new / 3.0),
        ],
        dtype="float64",
    )


def question_features(question: str) -> tuple[list[str], list[float]]:
    raw = str(question).strip()
    tokens = _TOKENS.findall(raw.lower())
    token_set = set(tokens)
    first = tokens[0] if tokens else ""
    names = [
        "question_token_count",
        "question_character_count",
        "question_numeric_fraction",
        "question_capitalized_token_count",
        "question_is_boolean",
        "question_is_comparison",
        "question_has_and",
        "question_has_or",
        *[f"question_starts_{word}" for word in _WH_WORDS],
    ]
    values = [
        min(1.0, len(tokens) / 32.0),
        min(1.0, len(raw) / 240.0),
        sum(token.isdigit() for token in tokens) / max(1, len(tokens)),
        min(1.0, sum(part[:1].isupper() for part in raw.split()[1:]) / 8.0),
        float(first in _BOOLEAN_STARTS),
        float(bool(token_set & _COMPARISON_TERMS)),
        float("and" in token_set),
        float("or" in token_set),
        *[float(first == word) for word in _WH_WORDS],
    ]
    return names, values


def evidence_features(question: str, action: dict, previous_ids: set[str]) -> tuple[list[str], list[float]]:
    question_tokens = words(question)
    evidence = tuple(action.get("candidate_evidence") or action.get("evidence") or ())
    ids = {str(item.get("content_id", "")) for item in evidence if item.get("content_id")}
    title_sets = [words(str(item.get("title", ""))) for item in evidence]
    title_union = set().union(*title_sets) if title_sets else set()
    scores = np.asarray([float(item.get("score", 0.0)) for item in evidence], dtype="float64")
    overlaps = [jaccard(question_tokens, title) for title in title_sets]
    title_hits = [float(bool(question_tokens & title)) for title in title_sets]
    names = [
        "evidence_count",
        "evidence_unique_title_fraction",
        "evidence_score_std",
        "evidence_score_range",
        "question_token_coverage_by_titles",
        "title_token_precision_for_question",
        "maximum_question_title_jaccard",
        "mean_question_title_jaccard",
        "title_question_hit_fraction",
        "evidence_new_id_fraction",
        "evidence_retained_id_fraction",
    ]
    values = [
        min(1.0, len(evidence) / 12.0),
        len({" ".join(sorted(title)) for title in title_sets}) / max(1, len(title_sets)),
        min(1.0, float(scores.std()) if len(scores) else 0.0),
        min(1.0, float(scores.max() - scores.min()) if len(scores) else 0.0),
        len(question_tokens & title_union) / max(1, len(question_tokens)),
        len(question_tokens & title_union) / max(1, len(title_union)),
        max(overlaps, default=0.0),
        float(np.mean(overlaps)) if overlaps else 0.0,
        float(np.mean(title_hits)) if title_hits else 0.0,
        len(ids - previous_ids) / max(1, len(ids)),
        len(ids & previous_ids) / max(1, len(previous_ids)),
    ]
    return names, values


def sequential_state(
    question: str,
    actions: Sequence[dict],
    step: int,
    manifest_names: tuple[str, ...] = CONTEXT_PREFIX_FEATURE_NAMES,
) -> np.ndarray:
    """Build the decision state after ``step + 1`` searches from prefixes only.

    Future retrieval blocks are zero-filled; only the current reader answer is
    observed.  Gold answers, gold hops and dataset identity never enter.
    """
    if not 0 <= step < DECISION_POINTS or len(actions) <= step:
        raise ValueError(f"결정 단계 {step}에 필요한 검색 기록이 없습니다.")
    retrieval_count = manifest_names.index("answer_is_unknown")
    evidence_count = len(evidence_features("", {}, set())[1])
    blocks: list[np.ndarray] = []
    previous_ids: set[str] = set()
    for action in actions[: step + 1]:
        policy_values = action["policy_features"]
        if len(policy_values) != len(manifest_names):
            raise ValueError("policy feature schema mismatch")
        _, evidence_values = evidence_features(question, action, previous_ids)
        blocks.append(np.asarray([*policy_values[:retrieval_count], *evidence_values], dtype="float64"))
        evidence = action.get("candidate_evidence") or action.get("evidence") or ()
        previous_ids = {
            str(item["content_id"]) for item in evidence if item.get("content_id")
        }
    current = actions[step]
    current_reader = [
        float(value)
        for name, value in zip(
            manifest_names[retrieval_count:], current["policy_features"][retrieval_count:], strict=True
        )
        if name not in READER_HISTORY_FEATURES
    ]
    structure = chain_structure_features(
        question, current.get("evidence_chain", ()), str(current.get("prediction", ""))
    )
    padding = np.zeros(retrieval_count + evidence_count, dtype="float64")
    _, question_values = question_features(question)
    return np.concatenate((
        *blocks,
        *([padding] * (DECISION_POINTS - step - 1)),
        np.asarray(question_values),
        np.asarray(current_reader),
        structure,
    ))


@dataclass(frozen=True)
class AnswerGainPolicy:
    """A trained ``HistGradientBoostingRegressor`` with calibrated thresholds."""

    model: object
    thresholds: tuple[float, ...]
    name: str

    @classmethod
    def from_path(cls, path: Path) -> "AnswerGainPolicy":
        import joblib

        # joblib uses pickle: load only the repository's own trusted models.
        bundle = joblib.load(path)
        if tuple(bundle.get("manifest_feature_names", ())) != CONTEXT_PREFIX_FEATURE_NAMES:
            raise ValueError("정책 모델의 입력 특징 구성이 현재 코드와 다릅니다.")
        thresholds = tuple(float(value) for value in bundle["thresholds"])
        if len(thresholds) != DECISION_POINTS:
            raise ValueError("정책 모델의 단계별 임곗값 수가 올바르지 않습니다.")
        return cls(bundle["model"], thresholds, Path(path).stem)

    def predict_gain(self, state: np.ndarray) -> float:
        return float(self.model.predict(np.asarray(state, dtype="float64").reshape(1, -1))[0])


@dataclass(frozen=True)
class StepwiseConfig:
    """Retrieval settings of the paper's trajectory generator."""

    initial_candidates: int = 15
    top_k: int = 5
    search_candidates: int = 8
    planner_context_characters: int = 4000
    anchor_slots: int = 2
    reader_source_cap: int = 8
    additions_per_step: int = 1
    context_character_budget: int = 6000


@dataclass(frozen=True)
class SearchStep:
    """One observed prefix and the policy decision taken on it."""

    total_searches: int
    query: str
    sources: tuple[tuple[Chunk, float], ...]
    added: tuple[Chunk, ...]
    prediction: str
    evidence_chain: tuple[str, ...]
    predicted_gain: float | None
    threshold: float | None
    decision: str
    seconds: float


@dataclass(frozen=True)
class StepwiseResult:
    question: str
    steps: tuple[SearchStep, ...]
    policy_name: str
    seconds: float

    @property
    def final(self) -> SearchStep:
        return self.steps[-1]

    @property
    def total_searches(self) -> int:
        return self.final.total_searches


@dataclass
class StepwiseSearcher:
    """Run initial retrieval, then SEARCH while predicted gain beats the threshold."""

    retriever: object
    rewriter: FrozenOllamaQueryRewriter
    reader: FrozenOllamaReader
    policy: AnswerGainPolicy
    config: StepwiseConfig = field(default_factory=StepwiseConfig)

    def run(self, question: str) -> StepwiseResult:
        config = self.config
        started = perf_counter()
        initial = self.retriever.search(question, top_k=config.initial_candidates)
        actions: list[dict] = []
        steps: list[SearchStep] = []
        previous_ids: set[str] = set()
        prior_predictions: list[str] = []
        prior_chains: list[tuple[str, ...]] = []
        for step in range(MAX_TOTAL_SEARCHES):
            step_started = perf_counter()
            planner_step = None
            query = question
            if step == 0:
                sources = list(initial[: config.top_k])
            else:
                # Replaying the planner up to this prefix is cheap: rewrites are
                # cached by prompt, so only the retrieval for new steps repeats.
                planned = run_planned_retrieval(
                    self.retriever,
                    self.rewriter,
                    question,
                    initial,
                    max_search_steps=step,
                    top_k=config.top_k,
                    search_candidates=config.search_candidates,
                    context_characters=config.planner_context_characters,
                    anchor_slots=config.anchor_slots,
                    reader_source_cap=config.reader_source_cap,
                    additions_per_step=config.additions_per_step,
                    replace_when_full=True,
                )
                sources = list(planned.sources_by_step[step])
                planner_step = planned.steps[step - 1]
                query = planner_step.rewrite.query
            retrieval = retrieval_features(step, sources, previous_ids, planner_step)
            context, provenance = evidence_context(sources, config.context_character_budget)
            result = self.reader.answer(question, context)
            chain = parse_evidence_chain(result.raw_response)
            reader = reader_sufficiency_features(
                question,
                result.prediction,
                chain,
                tuple({"title": chunk.title, "text": chunk.text} for chunk, _ in sources),
                prior_predictions,
                prior_chains,
            )
            actions.append({
                "policy_features": np.concatenate((retrieval, reader)).tolist(),
                "prediction": result.prediction,
                "evidence_chain": list(chain),
                "evidence": list(provenance),
                "candidate_evidence": [
                    {"content_id": chunk.source, "title": chunk.title, "score": float(score)}
                    for chunk, score in sources
                ],
            })
            gain = threshold = None
            if step < DECISION_POINTS:
                gain = self.policy.predict_gain(sequential_state(question, actions, step))
                threshold = self.policy.thresholds[step]
                decision = "SEARCH" if gain > threshold else "STOP"
            else:
                decision = "LIMIT"
            steps.append(SearchStep(
                total_searches=step + 1,
                query=query,
                sources=tuple(sources),
                added=tuple(planner_step.added_chunks) if planner_step is not None else (),
                prediction=result.prediction,
                evidence_chain=chain,
                predicted_gain=gain,
                threshold=threshold,
                decision=decision,
                seconds=perf_counter() - step_started,
            ))
            if decision != "SEARCH":
                break
            previous_ids = {chunk.source for chunk, _ in sources}
            prior_predictions.append(result.prediction)
            prior_chains.append(chain)
        return StepwiseResult(question, tuple(steps), self.policy.name, perf_counter() - started)
