"""Executable proof programs for evidence-grounded multi-hop QA.

The language model is restricted to compiling cited atomic facts.  A small
deterministic executor performs graph dependency checks and comparison logic,
so the final operation is not delegated back to the generator.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from time import perf_counter

import requests
from dateutil import parser as date_parser

from .qa_evaluation import AnswerCache, ReaderConfig, ReaderResult, normalize_answer


PROMPT_VERSION = "counterfactual_proof_program_v1_json"
CONTRACT_PROMPT_VERSION = "question_only_proof_contract_v2_json"
BINDING_PROMPT_VERSION = "immutable_proof_contract_binding_v2_json"
CACHE_FORMAT_VERSION = 1
OPERATIONS = ("lookup", "chain", "intersection", "compare", "boolean")
COMPARATORS = ("none", "earlier", "later", "larger", "smaller", "equal", "not_equal")
PROGRAM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": list(OPERATIONS)},
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 1, "maximum": 6},
                    "subject": {"type": "string", "maxLength": 120},
                    "relation": {"type": "string", "maxLength": 120},
                    "object": {"type": "string", "maxLength": 160},
                    "evidence_id": {"type": "integer", "minimum": 1, "maximum": 12},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1, "maximum": 6},
                        "maxItems": 4,
                    },
                },
                "required": ["id", "subject", "relation", "object", "evidence_id", "depends_on"],
                "additionalProperties": False,
            },
        },
        "answer_step": {"type": "integer", "minimum": 1, "maximum": 6},
        "answer_node": {"type": "string", "maxLength": 160},
        "comparison": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "enum": list(COMPARATORS)},
                "left_step": {"type": "integer", "minimum": 0, "maximum": 6},
                "left_entity": {"type": "string", "maxLength": 160},
                "left_value": {"type": "string", "maxLength": 160},
                "right_step": {"type": "integer", "minimum": 0, "maximum": 6},
                "right_entity": {"type": "string", "maxLength": 160},
                "right_value": {"type": "string", "maxLength": 160},
            },
            "required": [
                "operator", "left_step", "left_entity", "left_value",
                "right_step", "right_entity", "right_value",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["operation", "steps", "answer_step", "answer_node", "comparison"],
    "additionalProperties": False,
}
CONTRACT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": list(OPERATIONS)},
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 1, "maximum": 6},
                    "input": {"type": "string", "maxLength": 120},
                    "relation": {"type": "string", "maxLength": 120},
                    "output": {"type": "string", "maxLength": 120},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1, "maximum": 6},
                        "maxItems": 4,
                    },
                },
                "required": ["id", "input", "relation", "output", "depends_on"],
                "additionalProperties": False,
            },
        },
        "answer_step": {"type": "integer", "minimum": 1, "maximum": 6},
        "comparison": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "enum": list(COMPARATORS)},
                "left_step": {"type": "integer", "minimum": 0, "maximum": 6},
                "left_entity": {"type": "string", "maxLength": 160},
                "right_step": {"type": "integer", "minimum": 0, "maximum": 6},
                "right_entity": {"type": "string", "maxLength": 160},
            },
            "required": ["operator", "left_step", "left_entity", "right_step", "right_entity"],
            "additionalProperties": False,
        },
    },
    "required": ["operation", "steps", "answer_step", "comparison"],
    "additionalProperties": False,
}
BINDING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "bindings": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "integer", "minimum": 1, "maximum": 6},
                    "subject": {"type": "string", "maxLength": 120},
                    "object": {"type": "string", "maxLength": 160},
                    "evidence_id": {"type": "integer", "minimum": 1, "maximum": 12},
                },
                "required": ["step_id", "subject", "object", "evidence_id"],
                "additionalProperties": False,
            },
        },
        "answer_node": {"type": "string", "maxLength": 160},
    },
    "required": ["bindings", "answer_node"],
    "additionalProperties": False,
}
_SECTION = re.compile(r"(?ms)^\[(\d+)\][^\n]*\n(.*?)(?=\n\n\[\d+\]|\Z)")
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class ProofProgramResult:
    prediction: str
    operation: str
    valid: bool
    required_step_ids: tuple[int, ...]
    counterfactual_necessity: float
    distractor_invariant: bool
    raw_response: str
    cache_hit: bool
    latency_seconds: float


def proof_program_prompt(question: str, context: str) -> str:
    return f"""Compile the question and supplied evidence into a small executable proof program.
Do not answer from memory. Every step must be one atomic fact explicitly stated in its cited
[evidence_id]. Order bridge steps so later steps depend_on the earlier entity-producing steps.

Choose one operation:
- lookup: one directly supported fact; answer_step points to it.
- chain: follow relations; the executor returns the object of answer_step.
- intersection: two or more facts identify the same answer_node satisfying all constraints.
- compare: extract both scalar/date values; the executor, not you, chooses left_entity or right_entity.
- boolean: extract both values; the executor returns yes/no for equality or inequality.

For compare/boolean, set comparison operator and the step supplying each value. For other operations,
use operator=none, step=0 and empty comparison strings. Use exact entity/value wording from evidence.
Include only facts necessary for the shortest proof. If evidence is incomplete, use answer_node=unknown.
Return exactly one JSON object matching the required schema.

Question: {question}

Evidence:
{context}

Proof program:"""


def proof_contract_prompt(question: str) -> str:
    return f"""Compile only the logical structure of this multi-hop question before seeing any evidence.
Do not solve it and do not simplify away an intermediate relation. Use variables such as $person,
$place, $date, and $answer for unknown outputs. A nested phrase normally creates another step.

Examples:
Question: Where did the director of Film A die?
Contract: step 1 Film A --director--> $person; step 2 $person --place of death--> $answer,
with step 2 depending on step 1; operation=chain; answer_step=2.

Question: Which film has the older director, Film A or Film B?
Contract: Film A--director-->$p1; $p1--birth date-->$d1; Film B--director-->$p2;
$p2--birth date-->$d2; operation=compare; operator=earlier; left_step=2; right_step=4;
left_entity=Film A; right_entity=Film B.

Question: Are the directors of Film A and Film B from the same country?
Contract: two director steps followed by two country steps; operation=boolean; operator=equal.

Question: Which continent are both Magazine A and Magazine B published in?
Contract: Magazine A--publication continent-->$answer and Magazine B--publication continent-->$answer;
operation=intersection.

Use lookup only when the wording genuinely asks for one direct relation. Return exactly one JSON
object. For non-comparison operations use operator=none, comparison steps=0 and entity strings empty.

Question: {question}

Immutable proof contract:"""


def proof_binding_prompt(question: str, contract: dict, context: str) -> str:
    contract_text = json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
    return f"""Fill an immutable proof contract using only the numbered evidence.
The contract was created from the question before evidence was visible. Do not add, remove, reorder,
or reinterpret its steps. For every step, bind its input subject and output object to the exact entity,
date, number, or value stated in one cited [evidence_id]. Resolve a $variable using the output of the
step it depends on. If a slot cannot be supported, bind its object to unknown. answer_node is the
shortest final entity/value; for comparison it may be empty because the executor chooses the side.
Return exactly one JSON object.

Question: {question}

Immutable contract:
{contract_text}

Evidence:
{context}

Bindings:"""


def _validated_contract(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict) or payload.get("operation") not in OPERATIONS:
        return None
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 6:
        return None
    steps: dict[int, dict] = {}
    for raw in raw_steps:
        if not isinstance(raw, dict):
            return None
        try:
            step_id = int(raw["id"])
            dependencies = tuple(int(value) for value in raw.get("depends_on", ()))
        except (KeyError, TypeError, ValueError):
            return None
        if step_id in steps or any(value >= step_id or value < 1 for value in dependencies):
            return None
        if not all(str(raw.get(key, "")).strip() for key in ("input", "relation", "output")):
            return None
        steps[step_id] = {**raw, "id": step_id, "depends_on": list(dependencies)}
    if any(dependency not in steps for step in steps.values() for dependency in step["depends_on"]):
        return None
    try:
        answer_step = int(payload["answer_step"])
    except (KeyError, TypeError, ValueError):
        return None
    comparison = payload.get("comparison")
    if answer_step not in steps or not isinstance(comparison, dict):
        return None
    if payload["operation"] in {"compare", "boolean"}:
        try:
            comparison_steps = {int(comparison["left_step"]), int(comparison["right_step"])}
        except (KeyError, TypeError, ValueError):
            return None
        if not comparison_steps.issubset(steps):
            return None
    return {**payload, "steps": [steps[index] for index in sorted(steps)], "answer_step": answer_step}


def assemble_bound_program(contract: dict, bindings: dict) -> dict | None:
    """Combine immutable structure and evidence bindings; ignore structural edits."""
    validated = _validated_contract(contract)
    raw_bindings = bindings.get("bindings") if isinstance(bindings, dict) else None
    if validated is None or not isinstance(raw_bindings, list):
        return None
    binding_by_id = {}
    for raw in raw_bindings:
        if not isinstance(raw, dict):
            return None
        try:
            step_id = int(raw["step_id"])
            evidence_id = int(raw["evidence_id"])
        except (KeyError, TypeError, ValueError):
            return None
        if step_id in binding_by_id:
            return None
        binding_by_id[step_id] = {
            "subject": str(raw.get("subject", "")).strip(),
            "object": str(raw.get("object", "")).strip(),
            "evidence_id": evidence_id,
        }
    contract_steps = {int(step["id"]): step for step in validated["steps"]}
    if set(binding_by_id) != set(contract_steps):
        return None
    steps = []
    for step_id, contract_step in contract_steps.items():
        binding = binding_by_id[step_id]
        steps.append(
            {
                "id": step_id,
                "subject": binding["subject"],
                "relation": str(contract_step["relation"]),
                "object": binding["object"],
                "evidence_id": binding["evidence_id"],
                "depends_on": list(contract_step["depends_on"]),
            }
        )
    comparison_contract = validated["comparison"]
    left_step = int(comparison_contract.get("left_step", 0))
    right_step = int(comparison_contract.get("right_step", 0))
    comparison = {
        "operator": str(comparison_contract.get("operator", "none")),
        "left_step": left_step,
        "left_entity": str(comparison_contract.get("left_entity", "")),
        "left_value": binding_by_id[left_step]["object"] if left_step in binding_by_id else "",
        "right_step": right_step,
        "right_entity": str(comparison_contract.get("right_entity", "")),
        "right_value": binding_by_id[right_step]["object"] if right_step in binding_by_id else "",
    }
    return {
        "operation": validated["operation"],
        "steps": steps,
        "answer_step": validated["answer_step"],
        "answer_node": str(bindings.get("answer_node", "")).strip(),
        "comparison": comparison,
    }


def _parse_program(raw_response: str) -> dict | None:
    try:
        payload = json.loads(str(raw_response).strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sections(context: str) -> dict[int, str]:
    return {int(match.group(1)): match.group(2) for match in _SECTION.finditer(context)}


def _validated_steps(payload: dict, context: str) -> dict[int, dict] | None:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 6:
        return None
    sources = _sections(context)
    steps: dict[int, dict] = {}
    for raw in raw_steps:
        if not isinstance(raw, dict):
            return None
        try:
            step_id = int(raw["id"])
            evidence_id = int(raw["evidence_id"])
            dependencies = tuple(int(value) for value in raw.get("depends_on", ()))
        except (KeyError, TypeError, ValueError):
            return None
        subject = str(raw.get("subject", "")).strip()
        relation = str(raw.get("relation", "")).strip()
        obj = str(raw.get("object", "")).strip()
        if step_id in steps or not subject or not relation or not obj or evidence_id not in sources:
            return None
        if any(dependency >= step_id or dependency < 1 for dependency in dependencies):
            return None
        # A citation must lexically contain at least the subject or object.  It
        # is intentionally permissive for aliases but rejects fabricated IDs.
        cited = normalize_answer(sources[evidence_id])
        subject_norm = normalize_answer(subject)
        object_norm = normalize_answer(obj)
        if not (
            (subject_norm and subject_norm in cited)
            or (object_norm and object_norm in cited)
        ):
            return None
        steps[step_id] = {
            "id": step_id,
            "subject": subject,
            "relation": relation,
            "object": obj,
            "evidence_id": evidence_id,
            "depends_on": dependencies,
        }
    if any(dependency not in steps for step in steps.values() for dependency in step["depends_on"]):
        return None
    return steps


def _ancestors(step_ids: set[int], steps: dict[int, dict]) -> set[int]:
    required = set(step_ids)
    frontier = list(step_ids)
    while frontier:
        current = frontier.pop()
        for dependency in steps[current]["depends_on"]:
            if dependency not in required:
                required.add(dependency)
                frontier.append(dependency)
    return required


def _infer_comparator(question: str, proposed: str) -> str:
    lowered = normalize_answer(question)
    if any(phrase in lowered for phrase in ("same ", "both ", "share the same")):
        return "equal"
    if any(word in lowered.split() for word in ("earlier", "older", "first")):
        return "earlier"
    if any(phrase in lowered for phrase in ("more recently", "most recently")):
        return "later"
    if any(word in lowered.split() for word in ("later", "younger")):
        return "later"
    if any(word in lowered.split() for word in ("more", "larger", "greater", "higher", "most")):
        return "larger"
    if any(word in lowered.split() for word in ("less", "fewer", "smaller", "lower", "least")):
        return "smaller"
    return proposed if proposed in COMPARATORS else "none"


def _scalar(value: str) -> tuple[str, object] | None:
    text = str(value).strip()
    if not text:
        return None
    number = _NUMBER.search(text)
    month_or_date = bool(re.search(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b\d{4}\b",
        text,
        flags=re.IGNORECASE,
    ))
    if month_or_date:
        try:
            parsed = date_parser.parse(text, fuzzy=True, default=datetime(1, 1, 1))
            return "date", (parsed.year, parsed.month, parsed.day)
        except (ValueError, OverflowError):
            pass
    if number:
        try:
            return "number", float(number.group(0).replace(",", ""))
        except ValueError:
            pass
    normalized = normalize_answer(text)
    return ("text", normalized) if normalized else None


def execute_program(
    payload: dict,
    question: str,
    context: str,
    *,
    removed_steps: set[int] | None = None,
) -> tuple[str | None, tuple[int, ...]]:
    """Validate and execute a proof; removed_steps enables no-gold interventions."""
    operation = str(payload.get("operation", ""))
    if operation not in OPERATIONS:
        return None, ()
    steps = _validated_steps(payload, context)
    if not steps:
        return None, ()
    removed = removed_steps or set()
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        return None, ()

    if operation in {"compare", "boolean"}:
        try:
            left_step = int(comparison["left_step"])
            right_step = int(comparison["right_step"])
        except (KeyError, TypeError, ValueError):
            return None, ()
        if left_step not in steps or right_step not in steps:
            return None, ()
        required = _ancestors({left_step, right_step}, steps)
        if required & removed:
            return None, tuple(sorted(required))
        left = _scalar(str(comparison.get("left_value", "")))
        right = _scalar(str(comparison.get("right_value", "")))
        if left is None or right is None or left[0] != right[0]:
            return None, tuple(sorted(required))
        operator = _infer_comparator(question, str(comparison.get("operator", "none")))
        if operator in {"equal", "not_equal"}:
            equal = left[1] == right[1]
            return ("yes" if equal == (operator == "equal") else "no"), tuple(sorted(required))
        if operator not in {"earlier", "later", "larger", "smaller"} or left[1] == right[1]:
            return None, tuple(sorted(required))
        choose_left = left[1] < right[1] if operator in {"earlier", "smaller"} else left[1] > right[1]
        entity = comparison.get("left_entity") if choose_left else comparison.get("right_entity")
        prediction = str(entity or "").strip()
        return (prediction or None), tuple(sorted(required))

    try:
        answer_step = int(payload.get("answer_step"))
    except (TypeError, ValueError):
        return None, ()
    if answer_step not in steps:
        return None, ()
    required = _ancestors({answer_step}, steps)
    if operation == "intersection":
        required = set(steps)
    if required & removed:
        return None, tuple(sorted(required))
    if operation == "intersection":
        answer = str(payload.get("answer_node", "")).strip()
        normalized = normalize_answer(answer)
        appearances = sum(
            normalized in {normalize_answer(step["subject"]), normalize_answer(step["object"])}
            for step in steps.values()
        )
        return (answer if normalized and appearances >= 2 else None), tuple(sorted(required))
    return steps[answer_step]["object"], tuple(sorted(required))


def audit_counterfactuals(payload: dict, question: str, context: str) -> tuple[str | None, tuple[int, ...], float, bool]:
    prediction, required = execute_program(payload, question, context)
    if prediction is None or not required:
        return prediction, required, 0.0, False
    necessary = 0
    for step_id in required:
        ablated, _ = execute_program(payload, question, context, removed_steps={step_id})
        necessary += int(ablated is None or normalize_answer(ablated) != normalize_answer(prediction))
    cited_sources = {step["evidence_id"] for step in payload.get("steps", []) if isinstance(step, dict)}
    distractor_invariant = bool(set(_sections(context)) - cited_sources)
    return prediction, required, necessary / len(required), distractor_invariant


class FrozenProofProgramReader:
    def __init__(self, config: ReaderConfig, cache: AnswerCache, *, session: requests.Session | None = None) -> None:
        self.config = config
        self.cache = cache
        self.session = session or requests.Session()

    def answer(self, question: str, context: str) -> ProofProgramResult:
        prompt = proof_program_prompt(question, context)
        cache_key = self._cache_key(prompt)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._result(cached.raw_response, question, context, True, cached.latency_seconds)
        started = perf_counter()
        try:
            response = self.session.post(
                f"{self.config.host.rstrip('/')}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "format": PROGRAM_JSON_SCHEMA,
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
            response_payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(f"proof program reader 응답을 받지 못했습니다: {error}") from error
        raw_response = str(response_payload.get("response", "")).strip()
        latency = perf_counter() - started
        result = self._result(raw_response, question, context, False, latency)
        self.cache.put(
            cache_key,
            ReaderResult(
                raw_response=raw_response,
                prediction=result.prediction,
                cache_hit=False,
                latency_seconds=latency,
                prompt_eval_count=_optional_int(response_payload.get("prompt_eval_count")),
                eval_count=_optional_int(response_payload.get("eval_count")),
            ),
        )
        return result

    def _result(self, raw: str, question: str, context: str, cache_hit: bool, latency: float) -> ProofProgramResult:
        payload = _parse_program(raw)
        prediction, required, necessity, invariant = (
            audit_counterfactuals(payload, question, context) if payload else (None, (), 0.0, False)
        )
        operation = str(payload.get("operation", "invalid")) if payload else "invalid"
        return ProofProgramResult(
            prediction=prediction or "unknown",
            operation=operation,
            valid=prediction is not None,
            required_step_ids=required,
            counterfactual_necessity=necessity,
            distractor_invariant=invariant,
            raw_response=raw,
            cache_hit=cache_hit,
            latency_seconds=latency,
        )

    def _cache_key(self, prompt: str) -> str:
        encoded = json.dumps(
            {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "prompt_version": PROMPT_VERSION,
                "config": asdict(self.config),
                "prompt": prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class FrozenProofContractReader:
    """Question-only planning followed by immutable evidence binding."""

    def __init__(self, config: ReaderConfig, cache: AnswerCache, *, session: requests.Session | None = None) -> None:
        self.config = config
        self.cache = cache
        self.session = session or requests.Session()

    def answer(self, question: str, context: str) -> ProofProgramResult:
        started = perf_counter()
        contract_raw, contract_hit = self._generate(
            proof_contract_prompt(question),
            CONTRACT_JSON_SCHEMA,
            CONTRACT_PROMPT_VERSION,
        )
        contract = _validated_contract(_parse_program(contract_raw))
        if contract is None:
            return ProofProgramResult(
                prediction="unknown",
                operation="invalid_contract",
                valid=False,
                required_step_ids=(),
                counterfactual_necessity=0.0,
                distractor_invariant=False,
                raw_response=json.dumps({"contract": contract_raw}, ensure_ascii=False),
                cache_hit=contract_hit,
                latency_seconds=perf_counter() - started,
            )
        binding_raw, binding_hit = self._generate(
            proof_binding_prompt(question, contract, context),
            BINDING_JSON_SCHEMA,
            BINDING_PROMPT_VERSION,
        )
        bindings = _parse_program(binding_raw)
        program = assemble_bound_program(contract, bindings) if bindings else None
        prediction, required, necessity, invariant = (
            audit_counterfactuals(program, question, context) if program else (None, (), 0.0, False)
        )
        return ProofProgramResult(
            prediction=prediction or "unknown",
            operation=str(contract["operation"]),
            valid=prediction is not None,
            required_step_ids=required,
            counterfactual_necessity=necessity,
            distractor_invariant=invariant,
            raw_response=json.dumps(
                {"contract": contract, "bindings": bindings, "program": program},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            cache_hit=contract_hit and binding_hit,
            latency_seconds=perf_counter() - started,
        )

    def _generate(self, prompt: str, schema: dict, prompt_version: str) -> tuple[str, bool]:
        cache_key = self._cache_key(prompt, prompt_version)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached.raw_response, True
        started = perf_counter()
        try:
            response = self.session.post(
                f"{self.config.host.rstrip('/')}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "format": schema,
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
            raise RuntimeError(f"proof contract reader 응답을 받지 못했습니다: {error}") from error
        raw = str(payload.get("response", "")).strip()
        latency = perf_counter() - started
        self.cache.put(
            cache_key,
            ReaderResult(
                raw_response=raw,
                prediction="",
                cache_hit=False,
                latency_seconds=latency,
                prompt_eval_count=_optional_int(payload.get("prompt_eval_count")),
                eval_count=_optional_int(payload.get("eval_count")),
            ),
        )
        return raw, False

    def _cache_key(self, prompt: str, prompt_version: str) -> str:
        encoded = json.dumps(
            {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "prompt_version": prompt_version,
                "config": asdict(self.config),
                "prompt": prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
