"""Measure the study app's quantitative targets on the indexed NCS modules.

Ground truth comes from the documents' own structure (src/ncs_structure):
for every unit, passages that are *not* the app's unit anchors are turned into
learner questions by the frozen LLM.  The script then measures

* unit mapping accuracy (question -> curriculum unit, exact and module level);
* evidence accuracy (does the final evidence contain a passage of that unit);
* end-to-end latency of stepwise (paper policy) and single-search answers;
* agent success: grounded quiz and summary generation for every unit.

Questions are LLM-generated, so they read more like the source than real
learner questions would.  Nothing is written to the learner's study log.

Run from the repository root:  python -m scripts.measure_study_app
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.ncs_structure import label_chunks
from src.assistant import LocalStudyAssistant, build_answer_prompt
from src.curriculum import all_units, module_name

QUESTION_PROMPT = """당신은 NCS '{module}' 과목을 공부하는 자격증 수험생입니다.
아래 교재 구절을 읽고, 이 내용을 이해하기 위해 AI 튜터에게 물어볼 법한 질문을 한국어로 한 문장만 쓰세요.
교재 문장이나 단원 제목을 그대로 베끼지 말고 자연스러운 질문으로 쓰세요. 질문만 출력하세요.

{passage}
"""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-per-unit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-units", type=int, default=0, help="0 measures every unit; small values smoke-test")
    parser.add_argument("--mapping-only", type=Path, default=None,
                        help="Re-score unit mapping on the questions saved in this earlier report")
    parser.add_argument("--output", type=Path, default=Path("artifacts/measurements/study_app.json"))
    return parser.parse_args()


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype="float64")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _rate(flags: list[bool]) -> dict:
    return {"n": len(flags), "success": int(sum(flags)), "rate": float(np.mean(flags)) if flags else None}


def _map(index, item: dict) -> None:
    ranked = index.rank_units(item["question"])
    predicted, score = index.unit_for_text(item["question"])
    item.update({"predicted_unit": predicted, "mapping_score": score,
                 "top3": [unit_id for unit_id, _ in ranked[:3]]})


def _mapping_report(items: list[dict], modules: dict[str, str]) -> dict:
    return {
        "exact_unit": _rate([item["predicted_unit"] == item["unit_id"] for item in items]),
        "same_module": _rate([item["predicted_unit"] is not None
                              and modules[item["predicted_unit"]] == modules[item["unit_id"]] for item in items]),
        "top3_unit": _rate([item["unit_id"] in item["top3"] for item in items]),
        "unmatched": sum(item["predicted_unit"] is None for item in items),
    }


def main() -> None:
    args = _arguments()
    rng = random.Random(args.seed)
    assistant = LocalStudyAssistant(Path("data"), Path("data/index"))
    # Measurement questions must never enter the learner's study log.
    assistant.progress.save_question = lambda *unused: None
    if not assistant.ready_for_retrieval:
        raise SystemExit("인덱스가 없습니다. 웹앱에서 자료 인덱스를 먼저 생성하세요.")
    chunks = assistant.retriever.chunks
    labels = label_chunks([{"chunk_id": c.chunk_id, "source": c.source, "text": c.text} for c in chunks])
    index = assistant.unit_index
    units = all_units()
    names = {unit.unit_id: unit.name for unit in units}
    modules = {unit.unit_id: unit.module_id for unit in units}
    if args.max_units:
        units = rng.sample(units, args.max_units)
    by_unit: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        if labels.get(chunk.chunk_id) and len(chunk.text) >= 300:
            by_unit[labels[chunk.chunk_id]].append(chunk)

    # Warm up every model so the first measured item does not pay load time.
    assistant.answer("인공지능 플랫폼이란?")
    assistant._generate("안녕", max_tokens=4)

    if args.mapping_only:
        previous = json.loads(args.mapping_only.read_text(encoding="utf-8"))
        items = previous["items"]
        for item in items:
            _map(index, item)
        previous["mapping_before"] = previous.get("mapping_before", previous["mapping"])
        previous["mapping"] = _mapping_report(items, modules)
        previous["items"] = items
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"before": previous["mapping_before"], "after": previous["mapping"]},
                         ensure_ascii=False, indent=2))
        return

    started = time.perf_counter()
    items = []
    for unit in units:
        anchors = {chunk.chunk_id for chunk in index.anchors(unit.unit_id)}
        pool = [chunk for chunk in by_unit[unit.unit_id] if chunk.chunk_id not in anchors] or by_unit[unit.unit_id]
        for chunk in rng.sample(pool, min(args.questions_per_unit, len(pool))):
            raw = assistant._generate(
                QUESTION_PROMPT.format(module=module_name(unit.module_id), passage=chunk.text[:1500]), 120
            )
            question = re.sub(r"^\s*(질문\s*[:：]\s*)", "", raw.strip().splitlines()[0]).strip().strip('"')
            items.append({"unit_id": unit.unit_id, "source_chunk": chunk.chunk_id, "question": question})
    print(f"generated {len(items)} questions in {time.perf_counter() - started:.0f}s", flush=True)

    for number, item in enumerate(items, start=1):
        question = item["question"]
        _map(index, item)

        started = time.perf_counter()
        _, sources = assistant.answer(question)
        result = assistant.last_stepwise
        item["stepwise_seconds"] = time.perf_counter() - started
        item["stepwise_searches"] = result.total_searches if result else None
        item["stepwise_evidence_units"] = [labels.get(chunk.chunk_id) for chunk, _ in sources]
        item["stepwise_has_source_passage"] = any(chunk.chunk_id == item["source_chunk"] for chunk, _ in sources)

        started = time.perf_counter()
        fast_sources = assistant.search(question)
        context = assistant._evidence_context(fast_sources, 4800)
        assistant._generate(build_answer_prompt(question, context), max_tokens=420)
        item["single_seconds"] = time.perf_counter() - started
        item["single_evidence_units"] = [labels.get(chunk.chunk_id) for chunk, _ in fast_sources]
        if number % 10 == 0 or number == len(items):
            print(f"answered {number}/{len(items)}", flush=True)

    agent = []
    for unit in units:
        started = time.perf_counter()
        quiz, quiz_sources = assistant.make_quiz(unit.name, unit)
        quiz_seconds = time.perf_counter() - started
        quiz_ok = len(quiz.items) == 3 and all(1 <= item.evidence <= max(1, len(quiz_sources)) for item in quiz.items)
        started = time.perf_counter()
        summary, _ = assistant.summarize_unit(unit)
        summary_seconds = time.perf_counter() - started
        summary_ok = bool(summary.strip()) and "[근거" in summary and not summary.startswith("로컬 모델")
        agent.append({"unit_id": unit.unit_id, "quiz_items": len(quiz.items), "quiz_ok": quiz_ok,
                      "quiz_seconds": quiz_seconds, "summary_ok": summary_ok, "summary_seconds": summary_seconds,
                      "quiz_message": quiz.message})
        print(f"agent {unit.name}: quiz {len(quiz.items)} ok={quiz_ok} summary ok={summary_ok}", flush=True)

    def unit_hit(evidence, unit_id):
        return unit_id in evidence

    report = {
        "questions": len(items),
        "units": len(units),
        "question_source": "LLM-generated from structurally labeled NCS passages (excluding unit anchors)",
        "mapping": _mapping_report(items, modules),
        "evidence": {
            "stepwise_contains_unit": _rate([unit_hit(item["stepwise_evidence_units"], item["unit_id"]) for item in items]),
            "single_contains_unit": _rate([unit_hit(item["single_evidence_units"], item["unit_id"]) for item in items]),
            "stepwise_contains_source_passage": _rate([item["stepwise_has_source_passage"] for item in items]),
        },
        "latency_seconds": {
            "stepwise_answer": _summary([item["stepwise_seconds"] for item in items]),
            "single_search_answer": _summary([item["single_seconds"] for item in items]),
            "quiz": _summary([row["quiz_seconds"] for row in agent]),
            "summary": _summary([row["summary_seconds"] for row in agent]),
        },
        "stepwise_searches": _summary([item["stepwise_searches"] for item in items]),
        "agent": {
            "quiz_generation": _rate([row["quiz_ok"] for row in agent]),
            "summary_generation": _rate([row["summary_ok"] for row in agent]),
        },
        "items": [{**item, "unit_name": names[item["unit_id"]]} for item in items],
        "agent_items": agent,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"items", "agent_items"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
