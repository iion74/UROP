"""Deterministic answer generation and EM/F1 scoring for QA benchmarks.

This module is intentionally separate from the Korean MVP assistant.  It
freezes a short-answer prompt and Ollama decoding settings so retrieval policy
experiments change only the evidence supplied to the reader.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import string
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import requests

from .document_loader import Chunk


PROMPT_VERSION = "multihop_short_answer_v2_json"
EVIDENCE_CHAIN_PROMPT_VERSION = "multihop_evidence_chain_v3_json"
SUPPORTED_PROMPT_VERSIONS = (PROMPT_VERSION, EVIDENCE_CHAIN_PROMPT_VERSION)
CACHE_FORMAT_VERSION = 1
SPECIAL_ANSWERS = frozenset({"yes", "no", "noanswer"})
ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Only the shortest final answer span, entity, number, yes, no, or unknown.",
            "maxLength": 160,
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}
EVIDENCE_CHAIN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_chain": {
            "type": "array",
            "description": "Two to four short evidence clauses that connect the question to the answer.",
            "items": {"type": "string", "maxLength": 240},
            "minItems": 1,
            "maxItems": 4,
        },
        "answer": {
            "type": "string",
            "description": "Only the shortest final answer span, entity, number, yes, no, or unknown.",
            "maxLength": 160,
        },
    },
    "required": ["evidence_chain", "answer"],
    "additionalProperties": False,
}
_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_ANSWER_PREFIX = re.compile(r"^(?:final\s+answer|answer)\s*:\s*", flags=re.IGNORECASE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_TERM = re.compile(r"[a-z0-9]+")
_FOCUS_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
        "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
        "this", "to", "was", "were", "what", "when", "where", "which", "who",
        "whom", "whose", "why", "with",
    }
)


@dataclass(frozen=True)
class AnswerScore:
    em: float
    f1: float
    matched_gold: str


@dataclass(frozen=True)
class ReaderConfig:
    host: str
    model: str
    model_digest: str | None = None
    max_tokens: int = 64
    timeout_seconds: int = 300
    seed: int = 42
    temperature: float = 0.0
    keep_alive: str = "30m"
    prompt_version: str = PROMPT_VERSION
    thinking: bool = False


@dataclass(frozen=True)
class ReaderResult:
    raw_response: str
    prediction: str
    cache_hit: bool
    latency_seconds: float
    prompt_eval_count: int | None
    eval_count: int | None


def normalize_answer(text: str) -> str:
    """Apply the standard SQuAD/HotpotQA short-answer normalization."""
    lowered = str(text).lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(
    prediction: str,
    gold: str,
    *,
    special_answer_rule: bool = True,
) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    predicted_normalized = " ".join(predicted_tokens)
    gold_normalized = " ".join(gold_tokens)

    if special_answer_rule and (
        predicted_normalized in SPECIAL_ANSWERS or gold_normalized in SPECIAL_ANSWERS
    ) and predicted_normalized != gold_normalized:
        return 0.0
    if not predicted_tokens or not gold_tokens:
        # HotpotQA/2Wiki return zero when there is no overlapping token, even
        # when both normalized strings are empty. MuSiQue follows SQuAD and
        # treats two empty token sequences as a match.
        return float(predicted_tokens == gold_tokens) if not special_answer_rule else 0.0

    common = Counter(predicted_tokens) & Counter(gold_tokens)
    matches = sum(common.values())
    if matches == 0:
        return 0.0
    precision = matches / len(predicted_tokens)
    recall = matches / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def answer_score(
    prediction: str,
    answer: str,
    aliases: Iterable[str] = (),
    *,
    dataset: str = "hotpotqa",
) -> AnswerScore:
    golds = tuple(dict.fromkeys(gold for gold in (answer, *aliases) if str(gold).strip()))
    if not golds:
        raise ValueError("정답 또는 정답 alias가 필요합니다.")
    special_answer_rule = dataset in {"hotpotqa", "2wiki"}
    scored = [
        (
            exact_match(prediction, gold),
            token_f1(prediction, gold, special_answer_rule=special_answer_rule),
            gold,
        )
        for gold in golds
    ]
    em, f1, matched = max(scored, key=lambda item: (item[0], item[1]))
    return AnswerScore(em=em, f1=f1, matched_gold=matched)


def clean_prediction(raw_response: str) -> str:
    """Remove common wrapper noise without changing answer content."""
    text = str(raw_response).strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
        text = payload["answer"].strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = text.removeprefix("```").removesuffix("```").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        answer_lines = [line for line in lines if _ANSWER_PREFIX.match(line)]
        text = answer_lines[0] if answer_lines else lines[0]
    text = _ANSWER_PREFIX.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def evidence_context(
    sources: Sequence[tuple[Chunk, float]],
    character_budget: int,
    *,
    allocation: str = "sequential",
    compression: str = "none",
    focus_queries: Sequence[str] = (),
    focused_sentences_per_source: int = 2,
) -> tuple[str, tuple[dict[str, str | float], ...]]:
    """Build one fixed-budget reader context and return its provenance."""
    if character_budget < 1:
        raise ValueError("character_budget은 1 이상이어야 합니다.")
    if allocation not in {"sequential", "balanced"}:
        raise ValueError("allocation은 sequential 또는 balanced여야 합니다.")
    if compression not in {"none", "query_focused"}:
        raise ValueError("compression은 none 또는 query_focused여야 합니다.")
    if focused_sentences_per_source < 1:
        raise ValueError("focused_sentences_per_source는 1 이상이어야 합니다.")

    visible_sources = sources
    if compression == "query_focused":
        queries = tuple(query for query in focus_queries if str(query).strip())
        visible_sources = tuple(
            (
                Chunk(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=_query_focused_excerpt(
                        chunk.text,
                        queries,
                        focused_sentences_per_source,
                    ),
                    title=chunk.title,
                ),
                score,
            )
            for chunk, score in sources
        )

    if allocation == "balanced":
        return _balanced_evidence_context(visible_sources, character_budget)

    sections: list[str] = []
    provenance: list[dict[str, str | float]] = []
    remaining = character_budget
    for position, (chunk, score) in enumerate(visible_sources, start=1):
        header = f"[{position}] {chunk.title}\n"
        if remaining <= len(header):
            break
        text = chunk.text[: remaining - len(header)].rstrip()
        if not text:
            break
        sections.append(f"{header}{text}")
        provenance.append(
            {
                "chunk_id": chunk.chunk_id,
                "content_id": chunk.source,
                "title": chunk.title,
                "score": float(score),
            }
        )
        remaining -= len(header) + len(text) + 2
        if remaining <= 0:
            break
    return "\n\n".join(sections), tuple(provenance)


def _query_focused_excerpt(
    text: str,
    queries: Sequence[str],
    sentence_count: int,
) -> str:
    """Keep the most query-relevant sentences while preserving text order."""
    sentences = [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]
    if len(sentences) <= sentence_count:
        return " ".join(sentences)
    query_terms = [
        {term for term in _TERM.findall(query.lower()) if term not in _FOCUS_STOPWORDS}
        for query in queries
    ]
    query_terms = [terms for terms in query_terms if terms]
    if not query_terms:
        return " ".join(sentences[:sentence_count])

    scored: list[tuple[float, int]] = []
    for index, sentence in enumerate(sentences):
        terms = set(_TERM.findall(sentence.lower()))
        score = max(
            (
                len(terms & focus) / max(1.0, len(focus) ** 0.5)
                for focus in query_terms
            ),
            default=0.0,
        )
        scored.append((score, index))
    selected = sorted(index for _, index in sorted(scored, reverse=True)[:sentence_count])
    return " ".join(sentences[index] for index in selected)


def _balanced_evidence_context(
    sources: Sequence[tuple[Chunk, float]],
    character_budget: int,
) -> tuple[str, tuple[dict[str, str | float], ...]]:
    """Give every visible source a fair excerpt, redistributing unused space.

    The previous sequential allocator could spend the entire reader budget on
    early documents and silently hide newly retrieved evidence.  This
    water-filling allocation preserves source order while sharing the same
    total character budget across every source whose header and one text
    character fit.
    """
    selected: list[tuple[Chunk, float, str, str]] = []
    header_characters = 0
    for chunk, score in sources:
        text = chunk.text.strip()
        if not text:
            continue
        position = len(selected) + 1
        header = f"[{position}] {chunk.title}\n"
        candidate_count = position
        separator_characters = 2 * max(0, candidate_count - 1)
        minimum_text_characters = candidate_count
        if (
            header_characters
            + len(header)
            + separator_characters
            + minimum_text_characters
            > character_budget
        ):
            break
        selected.append((chunk, score, header, text))
        header_characters += len(header)

    if not selected:
        return "", ()

    text_budget = (
        character_budget
        - header_characters
        - 2 * max(0, len(selected) - 1)
    )
    allocations = [0] * len(selected)
    active = set(range(len(selected)))
    remaining = text_budget
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for index in tuple(sorted(active)):
            room = len(selected[index][3]) - allocations[index]
            if room <= 0:
                active.discard(index)
                continue
            given = min(room, share, remaining)
            allocations[index] += given
            remaining -= given
            progressed = progressed or given > 0
            if allocations[index] >= len(selected[index][3]):
                active.discard(index)
            if remaining <= 0:
                break
        if not progressed:
            break

    sections: list[str] = []
    provenance: list[dict[str, str | float]] = []
    for (chunk, score, header, text), allocated in zip(selected, allocations, strict=True):
        excerpt = text[:allocated].rstrip()
        if not excerpt:
            continue
        sections.append(f"{header}{excerpt}")
        provenance.append(
            {
                "chunk_id": chunk.chunk_id,
                "content_id": chunk.source,
                "title": chunk.title,
                "score": float(score),
            }
        )
    return "\n\n".join(sections), tuple(provenance)


def reader_prompt(
    question: str,
    context: str,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    if prompt_version not in SUPPORTED_PROMPT_VERSIONS:
        raise ValueError(f"지원하지 않는 reader prompt version입니다: {prompt_version}")
    if prompt_version == EVIDENCE_CHAIN_PROMPT_VERSION:
        return f"""Solve the multi-hop question using only the evidence below.
First identify the shortest chain of explicit evidence clauses needed to answer. Resolve every
entity reference and comparison against those clauses, and reject clauses about a similarly named
but different entity. Then return exactly one JSON object in this form:
{{"evidence_chain":["short fact 1","short fact 2"],"answer":"shortest final answer"}}
Use one to four short evidence clauses. The answer value must contain only the shortest answer span,
entity, number, yes, no, or unknown. Use unknown when any required link is absent. Do not include
citations or an explanation in the answer value.

Question: {question}

Evidence:
{context}

Answer:"""
    return f"""Solve the multi-hop question using only the evidence below.
Connect facts across documents and perform required comparisons internally.
If the evidence contains enough facts to infer the answer, make that inference.
Return exactly one JSON object in this form: {{"answer":"shortest final answer"}}
The answer value must contain only the shortest answer span, entity, number, yes, no, or unknown.
Never put an explanation, reasoning, citation, or full sentence in the answer value.
Use unknown only when a required fact is absent from the evidence.

Question: {question}

Evidence:
{context}

Answer:"""


class AnswerCache:
    """Persistent prompt-response cache for expensive counterfactual actions."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_cache (
                cache_key TEXT PRIMARY KEY,
                raw_response TEXT NOT NULL,
                prediction TEXT NOT NULL,
                latency_seconds REAL NOT NULL,
                prompt_eval_count INTEGER,
                eval_count INTEGER
            )
            """
        )
        self.connection.commit()

    def get(self, cache_key: str) -> ReaderResult | None:
        row = self.connection.execute(
            """
            SELECT raw_response, prediction, latency_seconds, prompt_eval_count, eval_count
            FROM answer_cache WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return ReaderResult(
            raw_response=str(row[0]),
            prediction=str(row[1]),
            cache_hit=True,
            latency_seconds=float(row[2]),
            prompt_eval_count=int(row[3]) if row[3] is not None else None,
            eval_count=int(row[4]) if row[4] is not None else None,
        )

    def put(self, cache_key: str, result: ReaderResult) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO answer_cache (
                cache_key, raw_response, prediction, latency_seconds, prompt_eval_count, eval_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                result.raw_response,
                result.prediction,
                result.latency_seconds,
                result.prompt_eval_count,
                result.eval_count,
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "AnswerCache":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


class FrozenOllamaReader:
    """A deterministic Ollama reader whose only varying input is evidence."""

    def __init__(
        self,
        config: ReaderConfig,
        cache: AnswerCache,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self.session = session or requests.Session()

    def answer(self, question: str, context: str) -> ReaderResult:
        prompt = reader_prompt(question, context, self.config.prompt_version)
        cache_key = self._cache_key(prompt)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        started = perf_counter()
        try:
            response = self.session.post(
                f"{self.config.host.rstrip('/')}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "format": (
                        EVIDENCE_CHAIN_JSON_SCHEMA
                        if self.config.prompt_version == EVIDENCE_CHAIN_PROMPT_VERSION
                        else ANSWER_JSON_SCHEMA
                    ),
                    "stream": False,
                    "think": self.config.thinking,
                    "keep_alive": self.config.keep_alive,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                        "seed": self.config.seed,
                    },
                },
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(
                f"동결 QA reader 응답을 받지 못했습니다: {error}. Ollama 실행 상태와 모델을 확인하세요."
            ) from error

        raw_response = str(payload.get("response", "")).strip()
        result = ReaderResult(
            raw_response=raw_response,
            prediction=clean_prediction(raw_response),
            cache_hit=False,
            latency_seconds=perf_counter() - started,
            prompt_eval_count=_optional_int(payload.get("prompt_eval_count")),
            eval_count=_optional_int(payload.get("eval_count")),
        )
        self.cache.put(cache_key, result)
        return result

    def _cache_key(self, prompt: str) -> str:
        reader_config = asdict(self.config)
        # ``thinking`` was added after the original 3,000-row reader cache was
        # created.  False is exactly the legacy behavior, so omit it to keep
        # those deterministic answers reusable.  True remains a distinct key.
        if not self.config.thinking:
            reader_config.pop("thinking", None)
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "reader": reader_config,
            "prompt": prompt,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def ollama_model_identity(
    host: str,
    model: str,
    *,
    session: requests.Session | None = None,
    timeout_seconds: int = 10,
) -> tuple[str, str | None]:
    """Resolve an Ollama tag to the exact installed name and digest."""
    client = session or requests.Session()
    try:
        response = client.get(f"{host.rstrip('/')}/api/tags", timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(f"Ollama 모델 목록을 확인하지 못했습니다: {error}") from error

    requested = {model, f"{model}:latest" if ":" not in model else model}
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", item.get("model", "")))
        short_name = name.removesuffix(":latest")
        if name in requested or short_name in requested or model == short_name:
            digest = str(item["digest"]) if item.get("digest") else None
            return name or model, digest
    available = ", ".join(
        str(item.get("name", item.get("model", "")))
        for item in payload.get("models", [])
        if isinstance(item, dict)
    )
    raise RuntimeError(f"Ollama에 reader 모델 '{model}'이 없습니다. 설치된 모델: {available or '없음'}")


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
