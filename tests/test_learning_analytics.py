import math
from datetime import date

from src.curriculum import all_units
from src.learning_analytics import build_report, retention


TODAY = date(2026, 10, 26)


def _unit(name: str):
    return next(unit for unit in all_units() if unit.name == name)


def test_retention_decays_and_grows_with_reviews():
    assert retention(0, 1, None) == 1.0
    assert retention(3, 1, None) == math.exp(-3 / 1.5)
    assert retention(3, 3, None) > retention(3, 1, None)
    assert retention(3, 1, 1.0) > retention(3, 1, 0.0)
    assert retention(3, 0, None) == 0.0


def test_report_maps_logs_to_coverage_mastery_and_exam_retention():
    unit = _unit("요구사항 명세화")
    asked = _unit("요구사항 수집")
    report = build_report(
        study_events=[("2026-10-20T10:00:00", unit.unit_id, "완료", 40, "식별자와 인수 기준")],
        question_events=[("2026-10-25T09:00:00", asked.unit_id, "요구사항은 어떻게 모아?"),
                         ("2026-10-25T09:05:00", None, "관련 없는 질문")],
        quiz_events=[("2026-10-21T10:00:00", unit.unit_id, 3, 3)],
        densities={unit.unit_id: 0.8},
        today=TODAY,
        exam_date=date(2026, 11, 25),
    )
    by_unit = report.by_unit()
    studied = by_unit[unit.unit_id]
    assert report.coverage == 2 / len(report.units)
    assert studied.mastery == (0.5 * 1.0 + 0.3 * 0.8 + 0.2 * 1.0) / 1.0
    assert studied.review_days == 2 and studied.last_studied == date(2026, 10, 21)
    assert 0 < studied.retention_exam < studied.retention_now < 1
    assert by_unit[asked.unit_id].mastery == 0.2  # question-only exposure
    assert by_unit[asked.unit_id].question_count == 1
    assert report.days_to_exam == 30
    assert report.daily_pace == report.remaining_units / 30


def test_untouched_units_have_zero_mastery_and_next_unit_is_recommended():
    report = build_report([], [], [], {}, TODAY)
    assert report.coverage == 0.0 and report.effective_knowledge == 0.0
    assert report.connectivity is None
    assert [item.title for item in report.recommendations] == ["다음 단원 시작"]
    assert report.recommendations[0].unit == all_units()[0]


def test_weak_units_and_missing_prerequisites_are_recommended():
    design = _unit("기능 설계")
    report = build_report(
        study_events=[("2026-10-01T10:00:00", design.unit_id, "복습 필요", 20, "기능")],
        question_events=[],
        quiz_events=[("2026-10-01T11:00:00", design.unit_id, 3, 1)],
        densities={design.unit_id: 0.2},
        today=TODAY,
    )
    titles = {(item.title, item.unit.module_id) for item in report.recommendations}
    assert ("퀴즈 재도전", "design") in titles
    assert ("핵심 요약 보기", "design") in titles
    # 03. 설계 requires 02. 요구사항 분석, which has not been studied at all.
    assert ("선수 단원 먼저", "requirements") in titles
    assert report.by_unit()[design.unit_id].prerequisites_ready is False
