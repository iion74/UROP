"""Observable reader-state features for adaptive proof stopping.

Gold answers and support annotations deliberately do not enter this module.
The features describe whether the frozen reader's current answer is stable and
whether its explicit evidence chain is actually reflected in the supplied
documents.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

import numpy as np

from .qa_evaluation import normalize_answer, token_f1


READER_SUFFICIENCY_FEATURE_NAMES = (
    "answer_is_unknown",
    "answer_is_boolean",
    "answer_token_count",
    "answer_in_question",
    "answer_document_fraction",
    "answer_title_fraction",
    "chain_is_empty",
    "chain_clause_count",
    "chain_average_document_support",
    "chain_minimum_document_support",
    "chain_answer_overlap",
    "chain_question_overlap",
    "previous_answer_exact",
    "previous_answer_f1",
    "initial_answer_exact",
    "initial_answer_f1",
    "answer_history_agreement",
    "previous_chain_jaccard",
    "evidence_document_count",
)
CHAIN_STRUCTURE_FEATURE_NAMES = (
    "chain_average_clause_length",
    "chain_explicit_fact_fraction",
    "chain_explicit_fact_count",
    "question_is_comparison",
    "comparison_explicit_branch_fraction",
    "comparison_branches_complete",
    "answer_in_explicit_clause",
    "answer_type_compatible",
)

_WORD = re.compile(r"[a-z0-9]+")
_COMPARISON = re.compile(
    r"\b(earlier|later|older|younger|longer|more|less|same|both|first|farther|closer)\b",
    flags=re.IGNORECASE,
)
_RELATION_TERMS = frozenset(
    {
        "are", "became", "borders", "born", "contains", "created", "did", "died",
        "directed", "does", "educated", "founded", "had", "has", "have", "is", "led",
        "located", "married", "occur", "occurred", "performed", "played", "plays", "recorded", "released",
        "was", "were",
    }
)
_BRANCH_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "at", "both", "did", "director", "directors", "do",
        "does", "born", "died", "death", "earlier", "film", "films", "first", "had", "has",
        "have", "in", "is", "later", "less", "lived", "longer", "more", "nationality", "of",
        "older", "on", "or", "people", "person", "same",
        "the", "to", "was", "were", "what", "when", "where", "which", "who", "younger",
    }
)


def parse_evidence_chain(raw_response: str) -> tuple[str, ...]:
    """Extract a structured evidence chain without trusting free-form text."""
    try:
        payload = json.loads(str(raw_response))
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(payload, dict) or not isinstance(payload.get("evidence_chain"), list):
        return ()
    return tuple(
        str(clause).strip()
        for clause in payload["evidence_chain"]
        if isinstance(clause, str) and str(clause).strip()
    )


def _tokens(text: str) -> set[str]:
    return set(normalize_answer(text).split())


def _contains(container: str, value: str) -> float:
    normalized_value = normalize_answer(value)
    if not normalized_value:
        return 0.0
    return float(normalized_value in normalize_answer(container))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _raw_tokens(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def _is_explicit_fact(clause: str) -> bool:
    terms = _raw_tokens(clause)
    return len(terms) >= 4 and bool(set(terms) & _RELATION_TERMS)


def _comparison_relation_match(question: str, clause: str) -> bool:
    question_terms = set(_raw_tokens(question))
    clause_terms = set(_raw_tokens(clause))
    has_number = bool(re.search(r"\d", str(clause)))
    if question_terms & {"born", "older", "younger"}:
        return "born" in clause_terms or has_number
    if question_terms & {"died", "death", "lived", "longer"}:
        return bool(clause_terms & {"died", "death"}) or has_number
    if question_terms & {"earlier", "later", "first", "more", "less"}:
        return has_number
    if question_terms & {"nationality", "country", "same"}:
        return bool(clause_terms & {"nationality", "country", "located", "born"})
    return True


def _answer_type_compatible(question: str, prediction: str) -> float:
    lowered = str(question).lower().strip()
    answer_terms = _raw_tokens(prediction)
    if lowered.startswith(("how many ", "how old ", "what year ")):
        return float(bool(re.search(r"\d", str(prediction))))
    if lowered.startswith("when "):
        months = {
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december",
        }
        return float(bool(re.search(r"\d", str(prediction))) or bool(set(answer_terms) & months))
    if "what does" in lowered and "stand for" in lowered:
        question_terms = set(_raw_tokens(question))
        return float(len(answer_terms) >= 2 and not set(answer_terms).issubset(question_terms))
    return 1.0


def chain_structure_features(
    question: str,
    evidence_chain: Sequence[str],
    prediction: str = "",
) -> np.ndarray:
    """Measure whether a chain contains relations, not just document titles.

    Comparative questions additionally require an explicit fact for both
    alternatives around ``or``.  This prevents a stable answer based on only
    one comparison branch from being treated as a completed proof.
    """
    clauses = tuple(str(clause) for clause in evidence_chain)
    explicit = tuple(_is_explicit_fact(clause) for clause in clauses)
    is_comparison = bool(_COMPARISON.search(str(question)))
    branch_fraction = 1.0
    branches_complete = 1.0
    if is_comparison:
        parts = re.split(r"\bor\b", str(question), flags=re.IGNORECASE)
        if len(parts) >= 2:
            branches = (
                set(_raw_tokens(parts[-2])[-8:]) - _BRANCH_STOPWORDS,
                set(_raw_tokens(parts[-1])[:8]) - _BRANCH_STOPWORDS,
            )
            covered = [
                bool(branch) and any(
                    is_fact
                    and bool(branch & set(_raw_tokens(clause)))
                    and _comparison_relation_match(question, clause)
                    for clause, is_fact in zip(clauses, explicit, strict=True)
                )
                for branch in branches
            ]
            branch_fraction = float(np.mean(covered))
            branches_complete = float(all(covered))
        else:
            explicit_count = sum(explicit)
            branch_fraction = min(1.0, explicit_count / 2.0)
            branches_complete = float(explicit_count >= 2)
    answer_tokens = set(_raw_tokens(prediction))
    answer_in_explicit = float(
        bool(answer_tokens)
        and any(
            is_fact and answer_tokens.issubset(set(_raw_tokens(clause)))
            for clause, is_fact in zip(clauses, explicit, strict=True)
        )
    )
    values = np.asarray(
        [
            min(1.0, np.mean([len(_raw_tokens(clause)) for clause in clauses]) / 20.0)
            if clauses
            else 0.0,
            float(np.mean(explicit)) if explicit else 0.0,
            min(1.0, sum(explicit) / 4.0),
            float(is_comparison),
            branch_fraction,
            branches_complete,
            answer_in_explicit,
            _answer_type_compatible(question, prediction),
        ],
        dtype="float64",
    )
    if len(values) != len(CHAIN_STRUCTURE_FEATURE_NAMES):
        raise RuntimeError("chain structure feature schema mismatch")
    return values


def reader_sufficiency_features(
    question: str,
    prediction: str,
    evidence_chain: Sequence[str],
    evidence_documents: Sequence[Mapping[str, str]],
    prior_predictions: Sequence[str] = (),
    prior_chains: Sequence[Sequence[str]] = (),
) -> np.ndarray:
    """Create gold-free answer stability and grounding features."""
    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = _tokens(prediction)
    documents = tuple(evidence_documents)
    document_texts = tuple(
        f"{document.get('title', '')} {document.get('text', '')}" for document in documents
    )
    titles = tuple(str(document.get("title", "")) for document in documents)
    answer_document_hits = sum(_contains(text, prediction) for text in document_texts)
    answer_title_hits = sum(_contains(title, prediction) for title in titles)

    clause_support = []
    document_token_sets = tuple(_tokens(text) for text in document_texts)
    for clause in evidence_chain:
        clause_tokens = _tokens(clause)
        clause_support.append(
            max(
                (
                    len(clause_tokens & document_tokens) / max(1, len(clause_tokens))
                    for document_tokens in document_token_sets
                ),
                default=0.0,
            )
        )
    chain_tokens = _tokens(" ".join(evidence_chain))
    question_tokens = _tokens(question)
    previous = str(prior_predictions[-1]) if prior_predictions else ""
    initial = str(prior_predictions[0]) if prior_predictions else ""
    history_matches = [
        float(normalize_answer(item) == normalized_prediction)
        for item in prior_predictions
        if normalize_answer(item)
    ]
    prior_chain_tokens = _tokens(" ".join(prior_chains[-1])) if prior_chains else set()

    values = np.asarray(
        [
            float(normalized_prediction in {"", "unknown", "noanswer"}),
            float(normalized_prediction in {"yes", "no"}),
            min(1.0, len(prediction_tokens) / 12.0),
            _contains(question, prediction),
            answer_document_hits / max(1, len(documents)),
            answer_title_hits / max(1, len(documents)),
            float(not evidence_chain),
            min(1.0, len(evidence_chain) / 4.0),
            float(np.mean(clause_support)) if clause_support else 0.0,
            min(clause_support, default=0.0),
            len(chain_tokens & prediction_tokens) / max(1, len(prediction_tokens)),
            len(chain_tokens & question_tokens) / max(1, len(question_tokens)),
            float(bool(prior_predictions) and normalize_answer(previous) == normalized_prediction),
            token_f1(prediction, previous, special_answer_rule=False) if prior_predictions else 0.0,
            float(bool(prior_predictions) and normalize_answer(initial) == normalized_prediction),
            token_f1(prediction, initial, special_answer_rule=False) if prior_predictions else 0.0,
            float(np.mean(history_matches)) if history_matches else 0.0,
            _jaccard(chain_tokens, prior_chain_tokens) if prior_chains else 0.0,
            min(1.0, len(documents) / 8.0),
        ],
        dtype="float64",
    )
    if len(values) != len(READER_SUFFICIENCY_FEATURE_NAMES):
        raise RuntimeError("reader sufficiency feature schema mismatch")
    return values
