"""Question-conditioned evidence bottleneck for multi-hop QA.

The selector is deliberately dataset agnostic.  It scores each retrieved
document from observable question/document/retrieval features and adds graph
features that describe how document titles connect the retrieved set.  Gold
support annotations are used only while training the selector.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np


TOKEN = re.compile(r"[a-z0-9]+")
QUESTION_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "both", "by", "did", "do",
        "does", "for", "from", "had", "has", "have", "how", "in", "is", "it",
        "of", "on", "or", "that", "the", "their", "to", "was", "were", "what",
        "when", "where", "which", "who", "whom", "whose", "with",
    }
)
FEATURE_NAMES = (
    "retrieval_score",
    "score_from_best",
    "score_from_mean",
    "reciprocal_rank",
    "question_title_jaccard",
    "question_text_recall",
    "question_title_recall",
    "title_exactly_in_question",
    "incoming_title_mentions",
    "outgoing_title_mentions",
    "max_incoming_retrieval_score",
    "text_length_log",
    "title_length_log",
    "question_is_comparison",
    "question_is_boolean",
)
POLICY_FEATURE_NAMES = (
    *(f"candidate_{name}" for name in FEATURE_NAMES),
    "action_is_answer",
    "action_is_already_selected",
    "candidate_links_from_selected",
    "candidate_links_to_selected",
    "candidate_novel_question_recall",
    "selected_fraction",
    "selected_question_coverage",
    "selected_internal_links",
    "selected_mean_retrieval_score",
    "remaining_max_retrieval_score",
)
WIDTH_FEATURE_NAMES = (
    *(f"support_probability_{index}" for index in range(1, 9)),
    *(f"retrieval_score_{index}" for index in range(1, 9)),
    *(f"probability_gap_{index}_{index + 1}" for index in range(1, 8)),
    "answer_action_probability",
    "probability_entropy",
    "question_token_count",
    "question_content_token_count",
    "question_capitalized_span_count",
    "question_possessive_count",
    "question_of_count",
    "question_relative_clause_count",
    "question_conjunction_count",
    "question_comma_count",
    "question_comparison",
    "question_boolean",
    "candidate_mean_internal_links",
    "candidate_max_internal_links",
)


def _tokens(text: str, *, remove_stopwords: bool = False) -> set[str]:
    terms = set(TOKEN.findall(str(text).lower()))
    return terms - QUESTION_STOPWORDS if remove_stopwords else terms


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(1.0, float(denominator))


def evidence_features(
    question: str,
    candidates: Sequence[Mapping],
    documents: Mapping[str, object],
) -> np.ndarray:
    """Return one feature row per candidate without gold or dataset features."""
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype="float64")
    question_terms = _tokens(question, remove_stopwords=True)
    lowered_question = " ".join(TOKEN.findall(question.lower()))
    scores = np.asarray([float(candidate.get("score", 0.0)) for candidate in candidates])
    best_score = float(scores.max())
    mean_score = float(scores.mean())

    materialized: list[tuple[str, str, set[str], set[str]]] = []
    for candidate in candidates:
        content_id = str(candidate.get("content_id", ""))
        document = documents.get(content_id)
        title = str(candidate.get("title", ""))
        text = ""
        if document is not None:
            title = str(getattr(document, "title", title))
            text = str(getattr(document, "text", ""))
        materialized.append((title, text, _tokens(title, remove_stopwords=True), _tokens(text)))

    rows: list[list[float]] = []
    for index, (candidate, (title, body, title_terms, body_terms)) in enumerate(
        zip(candidates, materialized, strict=True)
    ):
        incoming = 0
        outgoing = 0
        incoming_scores: list[float] = []
        normalized_body = f" {' '.join(TOKEN.findall(body.lower()))} "
        for other_index, (_, other_body, other_title_terms, _) in enumerate(materialized):
            if other_index == index or not title_terms or not other_title_terms:
                continue
            normalized_other_body = f" {' '.join(TOKEN.findall(other_body.lower()))} "
            title_phrase = " ".join(TOKEN.findall(title.lower()))
            other_title_phrase = " ".join(TOKEN.findall(materialized[other_index][0].lower()))
            if len(title_phrase) >= 4 and f" {title_phrase} " in normalized_other_body:
                incoming += 1
                incoming_scores.append(float(candidates[other_index].get("score", 0.0)))
            if len(other_title_phrase) >= 4 and f" {other_title_phrase} " in normalized_body:
                outgoing += 1

        overlap_title = question_terms & title_terms
        overlap_body = question_terms & body_terms
        score = float(candidate.get("score", 0.0))
        title_phrase = " ".join(TOKEN.findall(title.lower()))
        rows.append(
            [
                score,
                score - best_score,
                score - mean_score,
                1.0 / (index + 1),
                _safe_ratio(len(overlap_title), len(question_terms | title_terms)),
                _safe_ratio(len(overlap_body), len(question_terms)),
                _safe_ratio(len(overlap_title), len(question_terms)),
                float(bool(title_phrase) and title_phrase in lowered_question),
                float(incoming),
                float(outgoing),
                max(incoming_scores, default=0.0),
                float(np.log1p(len(body))),
                float(np.log1p(len(title))),
                float(any(word in lowered_question.split() for word in ("earlier", "later", "more", "less", "same"))),
                float(lowered_question.startswith(("are ", "do ", "does ", "did ", "is ", "was ", "were "))),
            ]
        )
    return np.asarray(rows, dtype="float64")


def select_evidence_indices(
    probabilities: Sequence[float],
    *,
    minimum: int = 2,
    maximum: int = 4,
    threshold: float = 0.35,
) -> tuple[int, ...]:
    """Select a compact variable-sized evidence set in original source order."""
    if minimum < 1 or maximum < minimum:
        raise ValueError("minimum/maximum evidence constraints are invalid")
    ranked = sorted(range(len(probabilities)), key=lambda index: (-float(probabilities[index]), index))
    selected = [index for index in ranked if float(probabilities[index]) >= threshold][:maximum]
    for index in ranked:
        if len(selected) >= min(minimum, len(ranked)):
            break
        if index not in selected:
            selected.append(index)
    return tuple(sorted(selected))


def proof_policy_features(
    question: str,
    candidates: Sequence[Mapping],
    documents: Mapping[str, object],
    selected: Sequence[int],
) -> np.ndarray:
    """Features for every document action followed by one ANSWER action.

    This representation makes document selection and proof execution compete
    in one learned action space.  It contains no answer, gold hop, dataset ID,
    or evaluation score.
    """
    base = evidence_features(question, candidates, documents)
    selected_set = {int(index) for index in selected if 0 <= int(index) < len(candidates)}
    question_terms = _tokens(question, remove_stopwords=True)
    titles: list[str] = []
    bodies: list[str] = []
    body_terms: list[set[str]] = []
    scores: list[float] = []
    for candidate in candidates:
        document = documents.get(str(candidate.get("content_id", "")))
        titles.append(str(getattr(document, "title", candidate.get("title", ""))))
        bodies.append(str(getattr(document, "text", "")))
        body_terms.append(_tokens(bodies[-1]))
        scores.append(float(candidate.get("score", 0.0)))

    selected_terms = set().union(*(body_terms[index] for index in selected_set)) if selected_set else set()
    selected_internal_links = 0
    for source_index in selected_set:
        normalized_body = f" {' '.join(TOKEN.findall(bodies[source_index].lower()))} "
        for target_index in selected_set:
            if source_index == target_index:
                continue
            title_phrase = " ".join(TOKEN.findall(titles[target_index].lower()))
            selected_internal_links += int(len(title_phrase) >= 4 and f" {title_phrase} " in normalized_body)

    common_state = [
        _safe_ratio(len(selected_set), len(candidates)),
        _safe_ratio(len(question_terms & selected_terms), len(question_terms)),
        float(selected_internal_links),
        float(np.mean([scores[index] for index in selected_set])) if selected_set else 0.0,
        max((score for index, score in enumerate(scores) if index not in selected_set), default=0.0),
    ]
    rows: list[list[float]] = []
    for index in range(len(candidates)):
        candidate_body = f" {' '.join(TOKEN.findall(bodies[index].lower()))} "
        links_from_selected = 0
        links_to_selected = 0
        for selected_index in selected_set:
            selected_body = f" {' '.join(TOKEN.findall(bodies[selected_index].lower()))} "
            candidate_title = " ".join(TOKEN.findall(titles[index].lower()))
            selected_title = " ".join(TOKEN.findall(titles[selected_index].lower()))
            links_from_selected += int(len(candidate_title) >= 4 and f" {candidate_title} " in selected_body)
            links_to_selected += int(len(selected_title) >= 4 and f" {selected_title} " in candidate_body)
        novel_terms = (question_terms & body_terms[index]) - selected_terms
        rows.append(
            [
                *base[index].tolist(),
                0.0,
                float(index in selected_set),
                float(links_from_selected),
                float(links_to_selected),
                _safe_ratio(len(novel_terms), len(question_terms)),
                *common_state,
            ]
        )
    rows.append([*([0.0] * len(FEATURE_NAMES)), 1.0, 0.0, 0.0, 0.0, 0.0, *common_state])
    return np.asarray(rows, dtype="float64")


def run_proof_completion_policy(
    model,
    question: str,
    candidates: Sequence[Mapping],
    documents: Mapping[str, object],
    *,
    max_documents: int = 6,
) -> tuple[tuple[int, ...], tuple[dict, ...]]:
    """Execute the learned document/ANSWER policy until ANSWER wins."""
    selected: list[int] = []
    trace: list[dict] = []
    for _ in range(min(max_documents, len(candidates)) + 1):
        features = proof_policy_features(question, candidates, documents, selected)
        probabilities = model.predict_proba(features)[:, 1]
        for index in selected:
            probabilities[index] = -1.0
        action = int(np.argmax(probabilities))
        answer_index = len(candidates)
        trace.append(
            {
                "action": "ANSWER" if action == answer_index else "RETRIEVE",
                "candidate_index": None if action == answer_index else action,
                "action_probability": float(probabilities[action]),
                "answer_probability": float(probabilities[answer_index]),
            }
        )
        if action == answer_index:
            break
        selected.append(action)
        if len(selected) >= max_documents:
            break
    return tuple(selected), tuple(trace)


def evidence_width_features(
    support_model,
    question: str,
    candidates: Sequence[Mapping],
    documents: Mapping[str, object],
) -> np.ndarray:
    """Observable features for predicting a question-specific proof width."""
    action_features = proof_policy_features(question, candidates, documents, ())
    action_probabilities = support_model.predict_proba(action_features)[:, 1]
    support_probabilities = sorted((float(value) for value in action_probabilities[:-1]), reverse=True)
    retrieval_scores = sorted(
        (float(candidate.get("score", 0.0)) for candidate in candidates), reverse=True
    )
    padded_probabilities = (support_probabilities + [0.0] * 8)[:8]
    padded_scores = (retrieval_scores + [0.0] * 8)[:8]
    gaps = [padded_probabilities[index] - padded_probabilities[index + 1] for index in range(7)]
    probability_sum = sum(padded_probabilities)
    distribution = [value / probability_sum for value in padded_probabilities] if probability_sum else [0.0] * 8
    entropy = -sum(value * np.log(value) for value in distribution if value > 0.0)
    raw_tokens = TOKEN.findall(str(question).lower())
    content_tokens = _tokens(question, remove_stopwords=True)
    capitalized_spans = re.findall(r"(?:\b[A-Z][\w'.-]*\b(?:\s+|$))+", str(question))
    lowered = " ".join(raw_tokens)
    incoming_index = FEATURE_NAMES.index("incoming_title_mentions")
    outgoing_index = FEATURE_NAMES.index("outgoing_title_mentions")
    base_features = evidence_features(question, candidates, documents)
    internal_links = base_features[:, incoming_index] + base_features[:, outgoing_index]
    return np.asarray(
        [
            *padded_probabilities,
            *padded_scores,
            *gaps,
            float(action_probabilities[-1]),
            float(entropy),
            float(len(raw_tokens)),
            float(len(content_tokens)),
            float(len(capitalized_spans)),
            float(str(question).count("'s")),
            float(raw_tokens.count("of")),
            float(sum(raw_tokens.count(word) for word in ("that", "where", "who", "whose", "which"))),
            float(sum(raw_tokens.count(word) for word in ("and", "or", "both"))),
            float(str(question).count(",")),
            float(any(word in lowered.split() for word in ("earlier", "later", "more", "less", "same", "older", "younger"))),
            float(lowered.startswith(("are ", "do ", "does ", "did ", "is ", "was ", "were "))),
            float(np.mean(internal_links)) if len(internal_links) else 0.0,
            float(np.max(internal_links)) if len(internal_links) else 0.0,
        ],
        dtype="float64",
    )


def select_topology_aware_indices(
    question: str,
    candidates: Sequence[Mapping],
    documents: Mapping[str, object],
    probabilities: Sequence[float],
    width: int,
) -> tuple[int, ...]:
    """Preserve question anchors and one outgoing proof edge per branch.

    Parallel comparison questions usually mention two anchor entities.  Pure
    pointwise ranking can spend the whole budget on one branch.  This selector
    reserves an anchor and a linked continuation for every visible branch,
    then fills the remaining evidence budget by learned support probability.
    """
    if width < 1:
        raise ValueError("width must be positive")
    materialized: list[tuple[str, str]] = []
    for candidate in candidates:
        document = documents.get(str(candidate.get("content_id", "")))
        materialized.append(
            (
                str(getattr(document, "title", candidate.get("title", ""))),
                str(getattr(document, "text", "")),
            )
        )
    normalized_question = f" {normalize_for_topology(question)} "
    anchors = []
    for index, (title, _) in enumerate(materialized):
        normalized_title = normalize_for_topology(title)
        if len(normalized_title) >= 4 and f" {normalized_title} " in normalized_question:
            anchors.append(index)
    anchors = sorted(anchors, key=lambda index: (-float(probabilities[index]), index))
    if len(anchors) < 2:
        anchors = []
    # A parallel proof needs one continuation per anchor.  Expand the width
    # structurally instead of using a dataset/type switch.
    # Reserve up to anchor -> bridge entity -> terminal value for each
    # parallel branch.  Shorter branches simply leave the remaining slots to
    # the learned ranker.
    effective_width = min(len(candidates), max(width, 3 * len(anchors)))
    selected: list[int] = []
    for anchor in anchors:
        if len(selected) >= effective_width:
            break
        if anchor not in selected:
            selected.append(anchor)
    for anchor in anchors:
        if len(selected) >= effective_width:
            break
        anchor_title = normalize_for_topology(materialized[anchor][0])
        anchor_body = f" {normalize_for_topology(materialized[anchor][1])} "
        linked = []
        for index, (title, body) in enumerate(materialized):
            if index == anchor or index in selected:
                continue
            title_phrase = normalize_for_topology(title)
            body_text = f" {normalize_for_topology(body)} "
            outgoing = len(title_phrase) >= 4 and f" {title_phrase} " in anchor_body
            incoming = len(anchor_title) >= 4 and f" {anchor_title} " in body_text
            if outgoing or incoming:
                linked.append(index)
        if linked:
            selected.append(max(linked, key=lambda index: (float(probabilities[index]), -index)))
    for index in sorted(range(len(candidates)), key=lambda item: (-float(probabilities[item]), item)):
        if len(selected) >= effective_width:
            break
        if index not in selected:
            selected.append(index)
    return tuple(selected)


def normalize_for_topology(text: str) -> str:
    return " ".join(TOKEN.findall(str(text).lower()))
