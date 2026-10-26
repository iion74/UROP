"""Learning-volume analytics over the NCS knowledge map.

The NCS learning-module structure (module → unit, with prerequisites) is the
standard knowledge graph.  Study notes, questions and quiz results are mapped
onto its units to measure:

* knowledge coverage: share of units the learner's text logs actually reached;
* semantic learning density: how well the learner's own notes match the
  official NCS source passages for that unit (cross-encoder relevance);
* retention: a simple forgetting-curve model whose stability grows with the
  number of review days and quiz accuracy, projected to the exam date.

``build_report`` is pure so it can be tested without loading any model.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from .curriculum import MODULES, LearningUnit, Module, all_units


STATUS_SCORE = {"완료": 1.0, "학습 중": 0.5, "복습 필요": 0.35}
# Exposure through questions alone is weaker evidence than a study log.
QUESTION_ONLY_SCORE = 0.2
BASE_STABILITY_DAYS = 1.5
MAX_STABILITY_DAYS = 120.0
MASTERED = 0.6
REVIEW_RETENTION = 0.6
LOW_DENSITY = 0.45
LOW_QUIZ_ACCURACY = 0.6
PREREQUISITE_READY = 0.3


@dataclass(frozen=True)
class UnitAnalytics:
    unit: LearningUnit
    status: str
    reached: bool
    density: float | None
    quiz_accuracy: float | None
    question_count: int
    review_days: int
    last_studied: date | None
    mastery: float
    retention_now: float
    retention_exam: float
    prerequisites_ready: bool


@dataclass(frozen=True)
class Recommendation:
    unit: LearningUnit
    action: str  # "summary" | "quiz" | "study"
    title: str
    reason: str
    priority: float


@dataclass(frozen=True)
class LearningReport:
    units: tuple[UnitAnalytics, ...]
    coverage: float
    mastery_coverage: float
    average_density: float | None
    connectivity: float | None
    effective_knowledge: float
    today: date
    exam_date: date | None
    recommendations: tuple[Recommendation, ...]

    @property
    def days_to_exam(self) -> int | None:
        return None if self.exam_date is None else max(0, (self.exam_date - self.today).days)

    @property
    def remaining_units(self) -> float:
        """Units still to secure for exam day, counting partial mastery."""
        return len(self.units) * (1.0 - self.effective_knowledge)

    @property
    def daily_pace(self) -> float | None:
        days = self.days_to_exam
        return None if not days else self.remaining_units / days

    def by_unit(self) -> dict[str, UnitAnalytics]:
        return {item.unit.unit_id: item for item in self.units}

    def module_mastery(self) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for item in self.units:
            grouped[item.unit.module_id].append(item.mastery)
        return {module_id: float(np.mean(values)) for module_id, values in grouped.items()}


def retention(days_elapsed: float, review_days: int, quiz_accuracy: float | None) -> float:
    """Forgetting curve ``exp(-t / S)`` whose stability doubles per review day."""
    if review_days < 1:
        return 0.0
    quality = 1.0 if quiz_accuracy is None else 0.6 + 0.8 * quiz_accuracy
    stability = min(MAX_STABILITY_DAYS, BASE_STABILITY_DAYS * 2 ** (review_days - 1) * quality)
    return math.exp(-max(0.0, days_elapsed) / stability)


def _day(timestamp: str) -> date:
    return datetime.fromisoformat(timestamp).date()


def build_report(
    study_events: Iterable[tuple[str, str, str, int, str]],
    question_events: Iterable[tuple[str, str | None, str]],
    quiz_events: Iterable[tuple[str, str | None, int, int]],
    densities: Mapping[str, float],
    today: date,
    exam_date: date | None = None,
    modules: tuple[Module, ...] = MODULES,
) -> LearningReport:
    units = [unit for unit in all_units() if any(module.module_id == unit.module_id for module in modules)]
    statuses: dict[str, str] = {}
    activity: dict[str, set[date]] = defaultdict(set)
    for created_at, unit_id, status, _, _ in study_events:
        statuses[unit_id] = status
        activity[unit_id].add(_day(created_at))
    questions: dict[str, int] = defaultdict(int)
    for created_at, unit_id, _ in question_events:
        if unit_id:
            questions[unit_id] += 1
            activity[unit_id].add(_day(created_at))
    quiz_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for created_at, unit_id, total, correct in quiz_events:
        if unit_id and total > 0:
            quiz_totals[unit_id][0] += total
            quiz_totals[unit_id][1] += correct
            activity[unit_id].add(_day(created_at))

    horizon = exam_date if exam_date and exam_date > today else today
    base: list[dict] = []
    for unit in units:
        reached = bool(activity.get(unit.unit_id))
        status = statuses.get(unit.unit_id, "미학습")
        density = densities.get(unit.unit_id)
        totals = quiz_totals.get(unit.unit_id)
        accuracy = totals[1] / totals[0] if totals else None
        components = [(0.5, STATUS_SCORE.get(status, QUESTION_ONLY_SCORE if reached else 0.0))]
        if density is not None:
            components.append((0.3, density))
        if accuracy is not None:
            components.append((0.2, accuracy))
        mastery = sum(weight * value for weight, value in components) / sum(weight for weight, _ in components)
        days = sorted(activity.get(unit.unit_id, ()))
        last = days[-1] if days else None
        base.append({
            "unit": unit,
            "status": status,
            "reached": reached,
            "density": density,
            "quiz_accuracy": accuracy,
            "question_count": questions.get(unit.unit_id, 0),
            "review_days": len(days),
            "last_studied": last,
            "mastery": mastery if reached else 0.0,
            "retention_now": retention((today - last).days, len(days), accuracy) if last else 0.0,
            "retention_exam": retention((horizon - last).days, len(days), accuracy) if last else 0.0,
        })

    module_values: dict[str, list[float]] = defaultdict(list)
    for item in base:
        module_values[item["unit"].module_id].append(item["mastery"])
    module_mastery = {module_id: float(np.mean(values)) for module_id, values in module_values.items()}
    prerequisites = {module.module_id: module.prerequisites for module in modules}
    analytics = tuple(
        UnitAnalytics(
            **item,
            prerequisites_ready=all(
                module_mastery.get(required, 0.0) >= PREREQUISITE_READY
                for required in prerequisites.get(item["unit"].module_id, ())
            ),
        )
        for item in base
    )
    reached = [item for item in analytics if item.reached]
    densities_seen = [item.density for item in reached if item.density is not None]
    return LearningReport(
        units=analytics,
        coverage=len(reached) / max(1, len(analytics)),
        mastery_coverage=sum(item.mastery >= MASTERED for item in analytics) / max(1, len(analytics)),
        average_density=float(np.mean(densities_seen)) if densities_seen else None,
        connectivity=(sum(item.prerequisites_ready for item in reached) / len(reached)) if reached else None,
        effective_knowledge=float(np.mean([item.mastery * item.retention_exam for item in analytics]))
        if analytics else 0.0,
        today=today,
        exam_date=exam_date,
        recommendations=_recommend(analytics),
    )


def _recommend(analytics: tuple[UnitAnalytics, ...], limit: int = 5) -> tuple[Recommendation, ...]:
    """Agentic next steps: weak or fading units first, then the next unit in order."""
    found: list[Recommendation] = []
    module_mastery: dict[str, list[float]] = defaultdict(list)
    for item in analytics:
        module_mastery[item.unit.module_id].append(item.mastery)
    prerequisites = {module.module_id: module.prerequisites for module in MODULES}
    for item in analytics:
        if not item.reached:
            continue
        if not item.prerequisites_ready:
            # Point to the first unmastered unit of the weakest prerequisite module.
            required = min(prerequisites.get(item.unit.module_id, ()),
                           key=lambda module_id: float(np.mean(module_mastery[module_id])))
            target = next((other for other in analytics
                           if other.unit.module_id == required and other.mastery < MASTERED), None)
            if target is not None:
                found.append(Recommendation(target.unit, "summary", "선수 단원 먼저",
                                            f"'{item.unit.name}'을(를) 학습 중이지만 선수 모듈 숙련도가 낮아 "
                                            "지식이 파편화될 수 있습니다.", 0.9))
        if item.quiz_accuracy is not None and item.quiz_accuracy < LOW_QUIZ_ACCURACY:
            found.append(Recommendation(item.unit, "quiz", "퀴즈 재도전",
                                        f"퀴즈 정답률 {item.quiz_accuracy:.0%}로 보강이 필요합니다.",
                                        1.0 - item.quiz_accuracy))
        if item.density is not None and item.density < LOW_DENSITY:
            found.append(Recommendation(item.unit, "summary", "핵심 요약 보기",
                                        f"메모와 NCS 원문의 의미 일치도가 {item.density:.0%}로 낮습니다.",
                                        0.8 * (1.0 - item.density)))
        if item.retention_now < REVIEW_RETENTION or item.status == "복습 필요":
            found.append(Recommendation(item.unit, "quiz", "복습 퀴즈",
                                        f"예상 기억 유지율이 {item.retention_now:.0%}로 떨어졌습니다.",
                                        0.7 * (1.0 - item.retention_now) + 0.1))
    next_unit = next(
        (item for item in analytics if not item.reached and item.prerequisites_ready), None
    )
    if next_unit is not None:
        found.append(Recommendation(next_unit.unit, "summary", "다음 단원 시작",
                                    "선수 학습이 준비된 다음 미학습 단원입니다.", 0.35))
    best: dict[tuple[str, str], Recommendation] = {}
    for item in found:
        key = (item.unit.unit_id, item.title)
        if key not in best or item.priority > best[key].priority:
            best[key] = item
    return tuple(sorted(best.values(), key=lambda item: item.priority, reverse=True)[:limit])


class SemanticUnitIndex:
    """Ground curriculum units in the indexed NCS passages.

    Each unit is anchored to the passages of its own module that best match
    the unit name.  Free text (a question) is mapped to the unit that is most
    similar on average to both the unit name and its anchors, because
    neighbouring units of one module often share passages; notes are scored
    against their unit's anchors with the cross-encoder so density is a
    calibrated 0-1 relevance.
    """

    def __init__(self, retriever, *, anchors_per_unit: int = 3, match_threshold: float = 0.80) -> None:
        self.retriever = retriever
        self.match_threshold = match_threshold
        self.units = all_units()
        self.anchor_indices: dict[str, list[int]] = {}
        self.anchor_vectors: dict[str, np.ndarray] = {}
        self.unit_vectors: dict[str, np.ndarray] = {}
        self._density_cache: dict[tuple[str, str], float] = {}
        if not retriever.ready:
            return
        vectors = retriever.index.reconstruct_n(0, len(retriever.chunks))
        modules = {module.module_id: module for module in MODULES}
        queries = [f"query: {modules[unit.module_id].name} {unit.name}" for unit in self.units]
        unit_vectors = np.asarray(retriever._model().encode(queries, normalize_embeddings=True), dtype="float32")
        for unit, unit_vector in zip(self.units, unit_vectors, strict=True):
            prefix = modules[unit.module_id].filename_prefix
            candidates = [index for index, chunk in enumerate(retriever.chunks) if prefix in chunk.source]
            if not candidates:
                continue
            scores = vectors[candidates] @ unit_vector
            best = [candidates[position] for position in np.argsort(scores)[::-1][:anchors_per_unit]]
            self.anchor_indices[unit.unit_id] = best
            self.anchor_vectors[unit.unit_id] = vectors[best]
            self.unit_vectors[unit.unit_id] = unit_vector

    @property
    def ready(self) -> bool:
        return bool(self.anchor_vectors)

    def anchors(self, unit_id: str):
        return [self.retriever.chunks[index] for index in self.anchor_indices.get(unit_id, [])]

    def unit_for_text(self, text: str) -> tuple[str | None, float]:
        if not self.ready or not text.strip():
            return None, 0.0
        vector = np.asarray(
            self.retriever._model().encode([f"query: {text}"], normalize_embeddings=True)[0], dtype="float32"
        )
        unit_id, score = max(
            (
                (unit_id, 0.5 * float((anchors @ vector).max()) + 0.5 * float(self.unit_vectors[unit_id] @ vector))
                for unit_id, anchors in self.anchor_vectors.items()
            ),
            key=lambda item: item[1],
        )
        return (unit_id if score >= self.match_threshold else None), score

    def densities(self, notes_by_unit: Mapping[str, list[str]]) -> dict[str, float]:
        """Mean over notes of the best cross-encoder relevance to the unit's NCS anchors."""
        result = {}
        for unit_id, notes in notes_by_unit.items():
            anchors = self.anchors(unit_id)
            notes = [note for note in notes if note.strip()]
            if not anchors or not notes:
                continue
            result[unit_id] = float(np.mean([self._note_density(unit_id, note, anchors) for note in notes]))
        return result

    def _note_density(self, unit_id: str, note: str, anchors) -> float:
        key = (unit_id, note)
        if key not in self._density_cache:
            ranked = self.retriever._rerank(note, [(chunk, 0.0) for chunk in anchors])
            self._density_cache[key] = max(score for _, score in ranked)
        return self._density_cache[key]


def notes_by_unit(study_events: Iterable[tuple[str, str, str, int, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for _, unit_id, _, _, note in study_events:
        if note and note.strip():
            grouped[unit_id].append(note.strip())
    return dict(grouped)


def unit_lookup() -> Callable[[str], LearningUnit | None]:
    units = {unit.unit_id: unit for unit in all_units()}
    return units.get
