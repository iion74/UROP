"""Learned local graph repair for incomplete multi-hop evidence paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .causal_evidence import TOKEN, _safe_ratio, _tokens, evidence_features, normalize_for_topology


REPAIR_FEATURE_NAMES = (
    *(f"proposal_{name}" for name in (
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
    )),
    "proposal_is_current_candidate",
    "proposal_max_graph_similarity",
    "proposal_seed_link_count",
    "proposal_max_seed_support_probability",
    "proposal_mean_seed_support_probability",
    "proposal_max_seed_retrieval_score",
    "proposal_title_mentioned_by_seed",
    "seed_title_mentioned_by_proposal",
    "proposal_question_coverage_gain",
    "selected_document_fraction",
    "current_support_probability_max",
    "current_support_probability_mean",
    "current_support_probability_entropy",
)
SYNDROME_FEATURE_NAMES = (
    *(f"repair_rank_score_{index}" for index in range(1, 6)),
    "repair_rank_gap_1_2",
    "repair_rank_entropy",
    "proposal_count_log",
    "selected_document_count",
    *(f"proposal_feature_mean_{name}" for name in REPAIR_FEATURE_NAMES),
    *(f"proposal_feature_max_{name}" for name in REPAIR_FEATURE_NAMES),
)


def build_graph_repair_proposals(
    retriever,
    candidates: Sequence[Mapping],
    selected: Sequence[int],
    *,
    neighbors_per_seed: int = 6,
    maximum_proposals: int = 48,
) -> list[dict]:
    """Return unselected originals plus graph neighbors with edge provenance."""
    if neighbors_per_seed < 1 or maximum_proposals < 1:
        raise ValueError("repair proposal limits must be positive")
    selected_set = {int(index) for index in selected}
    original_sources = {str(candidate.get("content_id", "")) for candidate in candidates}
    proposals: dict[str, dict] = {}
    for index, candidate in enumerate(candidates):
        if index in selected_set:
            continue
        source = str(candidate.get("content_id", ""))
        if not source:
            continue
        proposals[source] = {
            "chunk_id": str(candidate.get("chunk_id", "")),
            "content_id": source,
            "title": str(candidate.get("title", "")),
            "score": float(candidate.get("score", 0.0)),
            "is_current_candidate": 1.0,
            "max_graph_similarity": 1.0,
            "seed_indices": [index],
        }

    for seed_index, candidate in enumerate(candidates):
        chunk_position = retriever.chunk_positions.get(str(candidate.get("chunk_id", "")))
        if chunk_position is None:
            continue
        seed_score = float(candidate.get("score", 0.0))
        neighbors = sorted(
            retriever.graph.get(chunk_position, ()),
            key=lambda index: retriever._similarity(chunk_position, index),
            reverse=True,
        )[:neighbors_per_seed]
        for neighbor_position in neighbors:
            chunk = retriever.chunks[neighbor_position]
            if chunk.source in original_sources:
                continue
            similarity = float(
                np.clip((retriever._similarity(chunk_position, neighbor_position) + 1.0) / 2.0, 0.0, 1.0)
            )
            score = float(np.clip(0.55 * seed_score + 0.45 * similarity, 0.0, 1.0))
            current = proposals.get(chunk.source)
            if current is None:
                current = {
                    "chunk_id": chunk.chunk_id,
                    "content_id": chunk.source,
                    "title": chunk.title,
                    "score": score,
                    "is_current_candidate": 0.0,
                    "max_graph_similarity": similarity,
                    "seed_indices": [seed_index],
                }
                proposals[chunk.source] = current
            else:
                current["seed_indices"] = sorted(set((*current["seed_indices"], seed_index)))
                current["max_graph_similarity"] = max(
                    float(current["max_graph_similarity"]), similarity
                )
                if score > float(current["score"]):
                    current["chunk_id"] = chunk.chunk_id
                    current["title"] = chunk.title
                    current["score"] = score
    ranked = sorted(
        proposals.values(),
        key=lambda item: (
            -float(item["is_current_candidate"]),
            -float(item["score"]),
            -len(item["seed_indices"]),
            str(item["content_id"]),
        ),
    )
    return ranked[:maximum_proposals]


def graph_repair_features(
    question: str,
    candidates: Sequence[Mapping],
    selected: Sequence[int],
    support_probabilities: Sequence[float],
    proposals: Sequence[Mapping],
    documents: Mapping[str, object],
) -> np.ndarray:
    """Describe local repair edges using inference-observable features only."""
    if not proposals:
        return np.empty((0, len(REPAIR_FEATURE_NAMES)), dtype="float64")
    base = evidence_features(question, proposals, documents)
    selected_set = {int(index) for index in selected}
    selected_terms: set[str] = set()
    selected_materialized: list[tuple[str, str]] = []
    for index in selected_set:
        candidate = candidates[index]
        document = documents.get(str(candidate.get("content_id", "")))
        title = str(getattr(document, "title", candidate.get("title", "")))
        body = str(getattr(document, "text", ""))
        selected_materialized.append((title, body))
        selected_terms.update(_tokens(body))
    question_terms = _tokens(question, remove_stopwords=True)
    current_probabilities = np.asarray(support_probabilities, dtype="float64")
    probability_total = float(current_probabilities.sum())
    probability_distribution = (
        current_probabilities / probability_total
        if probability_total > 0.0
        else np.zeros_like(current_probabilities)
    )
    entropy = float(
        -sum(value * np.log(value) for value in probability_distribution if value > 0.0)
    )
    rows: list[list[float]] = []
    for position, proposal in enumerate(proposals):
        document = documents.get(str(proposal.get("content_id", "")))
        title = str(getattr(document, "title", proposal.get("title", "")))
        body = str(getattr(document, "text", ""))
        title_phrase = normalize_for_topology(title)
        body_phrase = f" {normalize_for_topology(body)} "
        seed_indices = [
            int(index)
            for index in proposal.get("seed_indices", ())
            if 0 <= int(index) < len(candidates)
        ]
        seed_probabilities = [float(support_probabilities[index]) for index in seed_indices]
        seed_scores = [float(candidates[index].get("score", 0.0)) for index in seed_indices]
        proposal_mentioned = 0
        seed_mentioned = 0
        for seed_index in seed_indices:
            seed = candidates[seed_index]
            seed_document = documents.get(str(seed.get("content_id", "")))
            seed_title = normalize_for_topology(
                str(getattr(seed_document, "title", seed.get("title", "")))
            )
            seed_body = f" {normalize_for_topology(str(getattr(seed_document, 'text', '')))} "
            proposal_mentioned += int(len(title_phrase) >= 4 and f" {title_phrase} " in seed_body)
            seed_mentioned += int(len(seed_title) >= 4 and f" {seed_title} " in body_phrase)
        proposal_terms = _tokens(body)
        coverage_gain = (question_terms & proposal_terms) - selected_terms
        rows.append(
            [
                *base[position].tolist(),
                float(proposal.get("is_current_candidate", 0.0)),
                float(proposal.get("max_graph_similarity", 0.0)),
                float(len(seed_indices)),
                max(seed_probabilities, default=0.0),
                float(np.mean(seed_probabilities)) if seed_probabilities else 0.0,
                max(seed_scores, default=0.0),
                float(proposal_mentioned),
                float(seed_mentioned),
                _safe_ratio(len(coverage_gain), len(question_terms)),
                _safe_ratio(len(selected_set), len(candidates)),
                float(current_probabilities.max()) if len(current_probabilities) else 0.0,
                float(current_probabilities.mean()) if len(current_probabilities) else 0.0,
                entropy,
            ]
        )
    return np.asarray(rows, dtype="float64")


def select_graph_repairs(
    probabilities: Sequence[float],
    proposals: Sequence[Mapping],
    *,
    threshold: float,
    maximum_additions: int = 2,
) -> tuple[int, ...]:
    """Choose a bounded number of distinct high-confidence repair documents."""
    if maximum_additions < 1 or not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid repair threshold or addition limit")
    ranked = sorted(range(len(proposals)), key=lambda index: (-float(probabilities[index]), index))
    selected: list[int] = []
    seen_sources: set[str] = set()
    for index in ranked:
        if float(probabilities[index]) < threshold or len(selected) >= maximum_additions:
            break
        source = str(proposals[index].get("content_id", ""))
        if source and source not in seen_sources:
            selected.append(index)
            seen_sources.add(source)
    return tuple(selected)


def repair_syndrome_features(
    proposal_features: np.ndarray,
    rank_scores: Sequence[float],
    selected_document_count: int,
) -> np.ndarray:
    """Summarize whether an evidence state contains a repairable missing edge."""
    scores = sorted((float(score) for score in rank_scores), reverse=True)
    padded = (scores + [0.0] * 5)[:5]
    score_total = sum(scores)
    distribution = [score / score_total for score in scores] if score_total else []
    entropy = -sum(value * np.log(value) for value in distribution if value > 0.0)
    if len(proposal_features):
        means = proposal_features.mean(axis=0).tolist()
        maxima = proposal_features.max(axis=0).tolist()
    else:
        means = [0.0] * len(REPAIR_FEATURE_NAMES)
        maxima = [0.0] * len(REPAIR_FEATURE_NAMES)
    result = np.asarray(
        [
            *padded,
            padded[0] - padded[1],
            float(entropy),
            float(np.log1p(len(proposal_features))),
            float(selected_document_count),
            *means,
            *maxima,
        ],
        dtype="float64",
    )
    if len(result) != len(SYNDROME_FEATURE_NAMES):
        raise RuntimeError("repair syndrome feature schema mismatch")
    return result
