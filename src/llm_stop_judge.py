"""Frozen Ollama evidence-sufficiency judge for adaptive retrieval.

The judge is a zero-shot comparison policy.  It is never trained and its
prompt never contains gold answers, supporting-fact annotations, hop counts,
or answer-level evaluation scores.
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


PROMPT_VERSION = "multihop_evidence_sufficiency_v1_json"
CACHE_FORMAT_VERSION = 1
DECISION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["STOP", "CONTINUE"],
            "description": "STOP only if the evidence supports a specific answer; otherwise CONTINUE.",
        },
        "supported_answer": {
            "type": "string",
            "description": "Shortest answer supported by the evidence, or an empty string when continuing.",
            "maxLength": 160,
        },
        "missing_information": {
            "type": "string",
            "description": "One factual relation still needed, or an empty string when stopping.",
            "maxLength": 320,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["decision", "supported_answer", "missing_information", "confidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class StopJudgeConfig:
    host: str
    model: str
    model_digest: str | None = None
    max_tokens: int = 120
    timeout_seconds: int = 300
    seed: int = 42
    temperature: float = 0.0
    keep_alive: str = "30m"
    thinking: bool = False
    prompt_version: str = PROMPT_VERSION


@dataclass(frozen=True)
class StopJudgeResult:
    decision: str
    supported_answer: str
    missing_information: str
    confidence: float
    raw_response: str
    valid: bool
    cache_hit: bool
    latency_seconds: float
    prompt_eval_count: int | None
    eval_count: int | None


def stop_judge_prompt(
    question: str,
    evidence: str,
    previous_queries: tuple[str, ...] = (),
) -> str:
    history = "\n".join(f"- {query}" for query in previous_queries) or "- none"
    return f"""You control whether a multi-hop QA system should retrieve more evidence.
Decide using only the original question and the currently visible evidence.
Return exactly one JSON object:
{{"decision":"STOP or CONTINUE","supported_answer":"short answer or empty",\
"missing_information":"one missing factual relation or empty","confidence":0.0}}

Decision rule:
- Decompose the question into all factual relations needed for the final answer.
- STOP only when the evidence explicitly supports every required relation, or supports a safe direct inference,
  and you can identify one specific shortest answer.
- CONTINUE when any required relation is absent, ambiguous, contradictory, or only guessed from prior knowledge.
- Use only the evidence. Do not rely on facts remembered from outside it.
- For STOP, fill supported_answer and leave missing_information empty.
- For CONTINUE, leave supported_answer empty and name exactly one missing relation.
- confidence is confidence in this STOP/CONTINUE decision, from 0 to 1.

Original question: {question}

Current evidence:
{evidence}

Previous search queries:
{history}

Decision:"""


def parse_stop_judgment(raw_response: str) -> tuple[str, str, str, float, bool]:
    """Parse a judge response; malformed output safely falls back to CONTINUE."""
    text = re.sub(
        r"<think>.*?</think>",
        "",
        str(raw_response),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return "CONTINUE", "", "invalid JSON response", 0.0, False
    if not isinstance(payload, dict):
        return "CONTINUE", "", "invalid JSON response", 0.0, False

    decision = str(payload.get("decision", "")).strip().upper()
    answer = str(payload.get("supported_answer", "")).strip()[:160]
    missing = str(payload.get("missing_information", "")).strip()[:320]
    try:
        confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    valid = decision in {"STOP", "CONTINUE"}
    if not valid:
        return "CONTINUE", "", missing or "invalid decision", confidence, False
    # A claimed STOP without an answer violates the grounded-decision prompt.
    # Continue conservatively instead of allowing malformed output to end search.
    if decision == "STOP" and (not answer or answer.casefold() in {"unknown", "noanswer"}):
        return "CONTINUE", "", missing or "no supported answer supplied", confidence, False
    return decision, answer, missing, confidence, True


class StopJudgeCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stop_judge_cache (
                cache_key TEXT PRIMARY KEY,
                raw_response TEXT NOT NULL,
                latency_seconds REAL NOT NULL,
                prompt_eval_count INTEGER,
                eval_count INTEGER
            )
            """
        )
        self.connection.commit()

    def get(self, cache_key: str) -> StopJudgeResult | None:
        row = self.connection.execute(
            """
            SELECT raw_response, latency_seconds, prompt_eval_count, eval_count
            FROM stop_judge_cache WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        decision, answer, missing, confidence, valid = parse_stop_judgment(str(row[0]))
        return StopJudgeResult(
            decision=decision,
            supported_answer=answer,
            missing_information=missing,
            confidence=confidence,
            raw_response=str(row[0]),
            valid=valid,
            cache_hit=True,
            latency_seconds=float(row[1]),
            prompt_eval_count=_optional_int(row[2]),
            eval_count=_optional_int(row[3]),
        )

    def put(self, cache_key: str, result: StopJudgeResult) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO stop_judge_cache (
                cache_key, raw_response, latency_seconds, prompt_eval_count, eval_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                result.raw_response,
                result.latency_seconds,
                result.prompt_eval_count,
                result.eval_count,
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "StopJudgeCache":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


class FrozenOllamaStopJudge:
    def __init__(
        self,
        config: StopJudgeConfig,
        cache: StopJudgeCache,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self.session = session or requests.Session()

    def judge(
        self,
        question: str,
        evidence: str,
        previous_queries: tuple[str, ...] = (),
    ) -> StopJudgeResult:
        prompt = stop_judge_prompt(question, evidence, previous_queries)
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
                    "format": DECISION_JSON_SCHEMA,
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
                f"동결 STOP judge 응답을 받지 못했습니다: {error}. Ollama 상태를 확인하세요."
            ) from error

        raw_response = str(payload.get("response", "")).strip()
        decision, answer, missing, confidence, valid = parse_stop_judgment(raw_response)
        result = StopJudgeResult(
            decision=decision,
            supported_answer=answer,
            missing_information=missing,
            confidence=confidence,
            raw_response=raw_response,
            valid=valid,
            cache_hit=False,
            latency_seconds=perf_counter() - started,
            prompt_eval_count=_optional_int(payload.get("prompt_eval_count")),
            eval_count=_optional_int(payload.get("eval_count")),
        )
        self.cache.put(cache_key, result)
        return result

    def _cache_key(self, prompt: str) -> str:
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "judge": asdict(self.config),
            "prompt": prompt,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
