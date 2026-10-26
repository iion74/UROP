"""Deterministic draft verification and conservative evidence-grounded correction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from time import perf_counter

import requests

from .qa_evaluation import AnswerCache, ReaderConfig, ReaderResult, clean_prediction, normalize_answer


PROMPT_VERSION = "grounded_draft_verifier_v1_json"
ADJUDICATOR_PROMPT_VERSION = "independent_verification_answer_v3_json"
CACHE_FORMAT_VERSION = 1
VERIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verification_chain": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
            "minItems": 1,
            "maxItems": 4,
        },
        "decision": {"type": "string", "enum": ["keep", "correct"]},
        "answer": {"type": "string", "maxLength": 160},
    },
    "required": ["verification_chain", "decision", "answer"],
    "additionalProperties": False,
}
ADJUDICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "independent_chain": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
            "minItems": 1,
            "maxItems": 4,
        },
        "answer": {"type": "string", "maxLength": 160},
    },
    "required": ["independent_chain", "answer"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class VerificationResult:
    prediction: str
    decision: str
    verification_chain: tuple[str, ...]
    raw_response: str
    cache_hit: bool
    latency_seconds: float
    prompt_eval_count: int | None
    eval_count: int | None


@dataclass(frozen=True)
class AdjudicationResult:
    accept: bool
    independent_chain: tuple[str, ...]
    raw_response: str
    cache_hit: bool
    latency_seconds: float


def verifier_prompt(question: str, context: str, draft: str) -> str:
    return f"""Verify a draft answer to a multi-hop question using only the supplied evidence.
Apply the same conservative procedure to every question, regardless of dataset or question type.

1. Identify the shortest chain of evidence clauses required by the question.
2. Resolve entity references, dates, comparisons, and similarly named entities.
3. Keep the draft if that chain supports it.
4. Correct the draft only when the evidence explicitly proves a different shortest answer.

Default to keep when evidence is merely incomplete or ambiguous. Never replace a supported draft with
a stylistic variant. Return exactly one JSON object:
{{"verification_chain":["fact 1","fact 2"],"decision":"keep","answer":"draft copied exactly"}}
For decision=keep, copy the draft exactly. For decision=correct, return only the corrected shortest
answer span, entity, number, yes, no, or unknown.

Question: {question}

Draft answer: {draft}

Evidence:
{context}

Verification:"""


def adjudicator_prompt(question: str, context: str) -> str:
    return f"""Solve this multi-hop question independently using only the supplied evidence.
You are an independent verifier: no earlier draft or proposed correction is shown to you.
Apply the same procedure to every dataset and question type. Identify the shortest complete evidence
chain, resolve entity references, dates and requested relations, then give the shortest answer. For
a comparison, state both compared values in the chain before deriving the answer. If a required link
is absent, answer unknown. Return exactly one JSON object:
{{"independent_chain":["fact 1","fact 2"],"answer":"shortest answer"}}

Question: {question}

Evidence:
{context}

Independent adjudication:"""


def _parse_response(raw_response: str, draft: str) -> tuple[str, str, tuple[str, ...]]:
    try:
        payload = json.loads(str(raw_response).strip())
    except (json.JSONDecodeError, TypeError):
        return "keep", draft, ()
    if not isinstance(payload, dict):
        return "keep", draft, ()
    decision = str(payload.get("decision", "keep")).strip().lower()
    chain = payload.get("verification_chain")
    parsed_chain = tuple(str(item).strip()[:240] for item in chain if str(item).strip()) if isinstance(chain, list) else ()
    if decision != "correct":
        return "keep", draft, parsed_chain
    answer = payload.get("answer")
    corrected = str(answer).strip()[:160] if isinstance(answer, str) else ""
    if not corrected:
        return "keep", draft, parsed_chain
    return "correct", corrected, parsed_chain


class FrozenGroundedAnswerVerifier:
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

    def verify(self, question: str, context: str, draft: str) -> VerificationResult:
        prompt = verifier_prompt(question, context, draft)
        cache_key = self._cache_key(prompt)
        cached = self.cache.get(cache_key)
        if cached is not None:
            decision, prediction, chain = _parse_response(cached.raw_response, draft)
            return VerificationResult(
                prediction=prediction,
                decision=decision,
                verification_chain=chain,
                raw_response=cached.raw_response,
                cache_hit=True,
                latency_seconds=cached.latency_seconds,
                prompt_eval_count=cached.prompt_eval_count,
                eval_count=cached.eval_count,
            )

        started = perf_counter()
        try:
            response = self.session.post(
                f"{self.config.host.rstrip('/')}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "format": VERIFICATION_JSON_SCHEMA,
                    "stream": False,
                    "think": False,
                    "keep_alive": self.config.keep_alive,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": self.config.max_tokens,
                        "seed": self.config.seed,
                    },
                },
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(f"grounded answer verifier 응답을 받지 못했습니다: {error}") from error

        raw_response = str(payload.get("response", "")).strip()
        decision, prediction, chain = _parse_response(raw_response, draft)
        latency = perf_counter() - started
        prompt_count = _optional_int(payload.get("prompt_eval_count"))
        eval_count = _optional_int(payload.get("eval_count"))
        self.cache.put(
            cache_key,
            ReaderResult(
                raw_response=raw_response,
                prediction=clean_prediction(raw_response),
                cache_hit=False,
                latency_seconds=latency,
                prompt_eval_count=prompt_count,
                eval_count=eval_count,
            ),
        )
        return VerificationResult(
            prediction=prediction,
            decision=decision,
            verification_chain=chain,
            raw_response=raw_response,
            cache_hit=False,
            latency_seconds=latency,
            prompt_eval_count=prompt_count,
            eval_count=eval_count,
        )

    def _cache_key(self, prompt: str) -> str:
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "config": asdict(self.config),
            "prompt": prompt,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class FrozenCorrectionAdjudicator:
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

    def adjudicate(
        self,
        question: str,
        context: str,
        draft: str,
        correction: str,
    ) -> AdjudicationResult:
        prompt = adjudicator_prompt(question, context)
        cache_key = self._cache_key(prompt)
        cached = self.cache.get(cache_key)
        if cached is not None:
            independent_answer, chain = _parse_adjudication(cached.raw_response)
            accept = _independent_agrees(independent_answer, draft, correction)
            return AdjudicationResult(
                accept=accept,
                independent_chain=chain,
                raw_response=cached.raw_response,
                cache_hit=True,
                latency_seconds=cached.latency_seconds,
            )

        started = perf_counter()
        try:
            response = self.session.post(
                f"{self.config.host.rstrip('/')}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "format": ADJUDICATION_JSON_SCHEMA,
                    "stream": False,
                    "think": False,
                    "keep_alive": self.config.keep_alive,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": self.config.max_tokens,
                        "seed": self.config.seed,
                    },
                },
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(f"correction adjudicator 응답을 받지 못했습니다: {error}") from error

        raw_response = str(payload.get("response", "")).strip()
        independent_answer, chain = _parse_adjudication(raw_response)
        accept = _independent_agrees(independent_answer, draft, correction)
        latency = perf_counter() - started
        self.cache.put(
            cache_key,
            ReaderResult(
                raw_response=raw_response,
                prediction="accept" if accept else "reject",
                cache_hit=False,
                latency_seconds=latency,
                prompt_eval_count=_optional_int(payload.get("prompt_eval_count")),
                eval_count=_optional_int(payload.get("eval_count")),
            ),
        )
        return AdjudicationResult(
            accept=accept,
            independent_chain=chain,
            raw_response=raw_response,
            cache_hit=False,
            latency_seconds=latency,
        )

    def _cache_key(self, prompt: str) -> str:
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "prompt_version": ADJUDICATOR_PROMPT_VERSION,
            "config": asdict(self.config),
            "prompt": prompt,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _parse_adjudication(raw_response: str) -> tuple[str, tuple[str, ...]]:
    try:
        payload = json.loads(str(raw_response).strip())
    except (json.JSONDecodeError, TypeError):
        return "", ()
    if not isinstance(payload, dict):
        return "", ()
    chain = payload.get("independent_chain")
    parsed_chain = tuple(str(item).strip()[:240] for item in chain if str(item).strip()) if isinstance(chain, list) else ()
    answer = payload.get("answer")
    return (str(answer).strip()[:160] if isinstance(answer, str) else ""), parsed_chain


def _independent_agrees(answer: str, draft: str, correction: str) -> bool:
    normalized = normalize_answer(answer)
    return bool(normalized) and normalized == normalize_answer(correction) and normalized != normalize_answer(draft)


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
