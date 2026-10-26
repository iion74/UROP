"""Deterministic, cached query reformulation for variable-hop retrieval.

The reformulator is frozen infrastructure, not the learned policy.  It turns
the original question and currently retrieved evidence into one standalone
query for the next unresolved link.  Policy experiments can then learn when
calling this relatively expensive action is worthwhile.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import requests


LEGACY_PROMPT_VERSION = "multihop_next_search_query_v1_json"
GAP_PROMPT_VERSION = "multihop_gap_query_v2_json"
PROMPT_VERSION = GAP_PROMPT_VERSION
CACHE_FORMAT_VERSION = 2
LEGACY_QUERY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "One concise standalone search query for the next unresolved fact.",
            "maxLength": 320,
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}
QUERY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "One concise standalone search query for the next unresolved fact.",
            "maxLength": 320,
        },
        "sufficient": {
            "type": "boolean",
            "description": "Whether the evidence supports every relation needed to answer.",
        },
        "missing_information": {
            "type": "string",
            "description": "The earliest unsupported relation, or an empty string when sufficient.",
            "maxLength": 320,
        },
    },
    "required": ["query", "sufficient", "missing_information"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class QueryRewriterConfig:
    host: str
    model: str
    model_digest: str | None = None
    max_tokens: int = 80
    timeout_seconds: int = 300
    seed: int = 42
    temperature: float = 0.0
    keep_alive: str = "30m"
    prompt_version: str = PROMPT_VERSION


@dataclass(frozen=True)
class QueryRewriteResult:
    query: str
    raw_response: str
    cache_hit: bool
    latency_seconds: float
    prompt_eval_count: int | None
    eval_count: int | None
    sufficient: bool | None = None
    missing_information: str = ""


def query_rewriter_prompt(
    question: str,
    evidence: str,
    previous_queries: tuple[str, ...] = (),
    prompt_version: str = PROMPT_VERSION,
) -> str:
    history = "\n".join(f"- {query}" for query in previous_queries) or "- none"
    if prompt_version == LEGACY_PROMPT_VERSION:
        return f"""You reformulate search queries for multi-hop question answering.
Find the earliest unresolved relation in the original question after reading the evidence.
Return exactly one JSON object: {{"query":"one concise standalone search query"}}

Rules:
- Resolve placeholders and indirect references with entities or facts found in the evidence.
- Ask only for the next missing fact, not for an explanation.
- Do not answer the original final question.
- Do not repeat a previous query.
- The query must make sense without seeing this prompt or the evidence.

Original question: {question}

Evidence:
{evidence}

Previous search queries:
{history}

Next search query:"""
    return f"""You judge evidence gaps and reformulate queries for multi-hop question answering.
Break the original question into the factual relations required for an answer. Check which relations
are explicitly supported by the evidence, then identify the earliest unsupported relation.
Return exactly one JSON object:
{{"sufficient":false,"missing_information":"one unsupported relation","query":"one standalone search query"}}

Rules:
- Resolve placeholders and indirect references with entities or facts found in the evidence.
- Set sufficient=true only when every relation required by the original question is explicitly supported.
- If evidence is sufficient, use an empty missing_information and make query verify the least-supported claim.
- If evidence is insufficient, describe exactly one missing relation and query only for that relation.
- Ask only for the next missing fact, not for an explanation.
- Do not answer the original final question.
- Do not repeat a previous query.
- The query must make sense without seeing this prompt or the evidence.

Original question: {question}

Evidence:
{evidence}

Previous search queries:
{history}

Next search query:"""


class QueryRewriteCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS query_rewrite_cache (
                cache_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                raw_response TEXT NOT NULL,
                latency_seconds REAL NOT NULL,
                prompt_eval_count INTEGER,
                eval_count INTEGER
            )
            """
        )
        self.connection.commit()

    def get(self, cache_key: str) -> QueryRewriteResult | None:
        row = self.connection.execute(
            """
            SELECT query, raw_response, latency_seconds, prompt_eval_count, eval_count
            FROM query_rewrite_cache WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return QueryRewriteResult(
            query=str(row[0]),
            raw_response=str(row[1]),
            cache_hit=True,
            latency_seconds=float(row[2]),
            prompt_eval_count=int(row[3]) if row[3] is not None else None,
            eval_count=int(row[4]) if row[4] is not None else None,
            sufficient=_parse_rewrite_fields(str(row[1]))[0],
            missing_information=_parse_rewrite_fields(str(row[1]))[1],
        )

    def put(self, cache_key: str, result: QueryRewriteResult) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO query_rewrite_cache (
                cache_key, query, raw_response, latency_seconds, prompt_eval_count, eval_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                result.query,
                result.raw_response,
                result.latency_seconds,
                result.prompt_eval_count,
                result.eval_count,
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "QueryRewriteCache":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


class FrozenOllamaQueryRewriter:
    def __init__(
        self,
        config: QueryRewriterConfig,
        cache: QueryRewriteCache,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self.session = session or requests.Session()

    def rewrite(
        self,
        question: str,
        evidence: str,
        previous_queries: tuple[str, ...] = (),
    ) -> QueryRewriteResult:
        prompt = query_rewriter_prompt(
            question, evidence, previous_queries, self.config.prompt_version
        )
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
                        LEGACY_QUERY_JSON_SCHEMA
                        if self.config.prompt_version == LEGACY_PROMPT_VERSION
                        else QUERY_JSON_SCHEMA
                    ),
                    "stream": False,
                    "think": False,
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
                f"동결 검색어 재작성기 응답을 받지 못했습니다: {error}. Ollama 상태를 확인하세요."
            ) from error

        raw_response = str(payload.get("response", "")).strip()
        query = clean_search_query(raw_response)
        sufficient, missing_information = _parse_rewrite_fields(raw_response)
        if not query:
            query = question.strip()
        result = QueryRewriteResult(
            query=query,
            raw_response=raw_response,
            cache_hit=False,
            latency_seconds=perf_counter() - started,
            prompt_eval_count=_optional_int(payload.get("prompt_eval_count")),
            eval_count=_optional_int(payload.get("eval_count")),
            sufficient=sufficient,
            missing_information=missing_information,
        )
        self.cache.put(cache_key, result)
        return result

    def _cache_key(self, prompt: str) -> str:
        payload = {
            "cache_format_version": (
                1 if self.config.prompt_version == LEGACY_PROMPT_VERSION else CACHE_FORMAT_VERSION
            ),
            "rewriter": asdict(self.config),
            "prompt": prompt,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


def clean_search_query(raw_response: str) -> str:
    text = str(raw_response).strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("query"), str):
        text = payload["query"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = text.removeprefix("```").removesuffix("```").strip()
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_line = re.sub(r"^(?:query|search query)\s*:\s*", "", first_line, flags=re.IGNORECASE)
    return first_line.strip().strip('"\'')[:320]


def _parse_rewrite_fields(raw_response: str) -> tuple[bool | None, str]:
    try:
        payload = json.loads(str(raw_response).strip())
    except (json.JSONDecodeError, TypeError):
        return None, ""
    if not isinstance(payload, dict):
        return None, ""
    sufficient = payload.get("sufficient")
    parsed_sufficient = bool(sufficient) if isinstance(sufficient, bool) else None
    missing = payload.get("missing_information")
    return parsed_sufficient, str(missing).strip()[:320] if isinstance(missing, str) else ""


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
