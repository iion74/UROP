from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

import streamlit as st

from .assistant import LocalStudyAssistant, Quiz
from .curriculum import MODULES, LearningUnit, all_units, module_name
from .learning_analytics import LearningReport, UnitAnalytics
from .retriever import reranker_probability


STATUS_ICON = {"완료": "●", "복습 필요": "●", "학습 중": "●", "미학습": "○"}
STATUS_CLASS = {"완료": "done", "복습 필요": "review", "학습 중": "active", "미학습": "idle"}
PAGES = ["대시보드", "학습 기록", "지식 맵", "AI 학습 도우미", "확인 퀴즈"]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Manrope', 'Pretendard', sans-serif; }
        .stApp { background: #f6f7fb; color: #111827; }
        [data-testid="stSidebar"] { background: #101828; border-right: 0; }
        [data-testid="stSidebar"] * { color: #f8fafc; }
        [data-testid="stSidebar"] .stCaption { color: #98a2b3 !important; }
        [data-testid="stSidebar"] .stRadio label { border-radius: 10px; padding: 5px 8px; }
        [data-testid="stSidebar"] .stRadio label:hover { background: #1d2939; }
        /* Sidebar text is white, but its input fields keep a white box, so they need dark text. */
        [data-testid="stSidebar"] [data-testid="stDateInputField"], [data-testid="stSidebar"] [data-baseweb="input"], [data-testid="stSidebar"] [data-baseweb="select"] > div { background: #ffffff !important; border-color: #d0d5dd !important; }
        [data-testid="stSidebar"] [data-testid="stDateInputField"] *, [data-testid="stSidebar"] [data-baseweb="input"] *, [data-testid="stSidebar"] [data-baseweb="select"] * { color: #101828 !important; -webkit-text-fill-color: #101828 !important; }
        [data-testid="stSidebar"] [data-testid="stDateInput"] svg, [data-testid="stSidebar"] [data-baseweb="input"] svg { fill: #475467 !important; color: #475467 !important; }
        [data-testid="stSidebar"] code { background: #1d2939 !important; color: #c7d7fe !important; }
        h1 { font-weight: 800 !important; letter-spacing: -0.04em; }
        h2, h3 { letter-spacing: -0.025em; }
        .block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1280px; }
        .hero { background: linear-gradient(110deg, #101828 0%, #182c5b 60%, #3157a9 100%); border-radius: 22px; color: white; padding: 2.15rem 2.25rem; margin-bottom: 1.45rem; }
        .hero .eyebrow { color: #b7cdfd; font-family: 'DM Mono', monospace; font-size: .76rem; letter-spacing: .08em; text-transform: uppercase; }
        .hero h1 { color: white; font-size: 2rem; margin: .38rem 0 .45rem; }
        .hero p { color: #d9e3ff; margin: 0; font-size: .96rem; }
        .metric-card { border: 1px solid #e4e7ec; border-radius: 16px; padding: 1.05rem 1.1rem; background: #fff; min-height: 120px; }
        .metric-card .label { color: #667085; font-size: .78rem; font-weight: 700; }
        .metric-card .value { color: #101828; font-size: 1.85rem; font-weight: 800; letter-spacing: -.06em; margin: .3rem 0; }
        .metric-card .hint { color: #667085; font-size: .77rem; }
        .section-kicker { color: #475467; font-family: 'DM Mono', monospace; font-size: .72rem; font-weight: 500; letter-spacing: .07em; text-transform: uppercase; }
        .module-card { background: white; border: 1px solid #eaecf0; border-radius: 15px; padding: .95rem 1rem; margin-bottom: .6rem; }
        .module-card .module-name { font-weight: 750; color: #101828; font-size: .92rem; }
        .module-card .module-meta { color: #667085; font-size: .76rem; margin-top: .22rem; }
        .badge { border-radius: 99px; padding: .2rem .55rem; font-size: .71rem; font-weight: 700; display: inline-block; }
        .badge.done { background: #dcfae6; color: #067647; }
        .badge.review { background: #fef0c7; color: #b54708; }
        .badge.active { background: #dbeafe; color: #175cd3; }
        .badge.idle { background: #f2f4f7; color: #667085; }
        .badge.search { background: #dbeafe; color: #175cd3; }
        .badge.stop { background: #dcfae6; color: #067647; }
        .source-card { background: #f8fafc; border-left: 3px solid #7f9cff; border-radius: 0 10px 10px 0; padding: .85rem 1rem; margin: .55rem 0; }
        .empty-card { background: white; border: 1px dashed #d0d5dd; border-radius: 16px; padding: 1.3rem; color: #667085; }
        .step-card { background: white; border: 1px solid #e4e7ec; border-radius: 14px; padding: .8rem 1rem; margin: .45rem 0; }
        .step-card .step-head { font-weight: 750; color: #101828; font-size: .9rem; }
        .step-card .step-meta { color: #667085; font-size: .78rem; margin-top: .25rem; }
        .rec-card { background: white; border: 1px solid #eaecf0; border-left: 4px solid #3157a9; border-radius: 0 14px 14px 0; padding: .75rem .95rem; margin-bottom: .35rem; }
        .rec-card .rec-title { font-weight: 750; color: #101828; font-size: .9rem; }
        .rec-card .rec-reason { color: #667085; font-size: .77rem; margin-top: .2rem; }
        [data-testid="stForm"] { border: 1px solid #e4e7ec; border-radius: 16px; background: white; padding: 1.1rem; }
        .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button { background: #244b9b; border-color: #244b9b; color: #ffffff; border-radius: 9px; font-weight: 700; }
        .stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover { background: #173875; border-color: #173875; color: #ffffff; }
        .stButton > button[kind="primary"] *, [data-testid="stFormSubmitButton"] > button * { color: #ffffff; }
        .stButton > button[kind="secondary"] { background: #ffffff; border: 1px solid #d0d5dd; color: #101828; border-radius: 9px; font-weight: 600; }
        .stButton > button[kind="secondary"] * { color: #101828; }
        .stButton > button[kind="secondary"]:hover { background: #eef4ff; border-color: #3157a9; color: #173875; }
        .stButton > button[kind="secondary"]:hover * { color: #173875; }        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(title: str, caption: str, eyebrow: str = "NOVA · LOCAL AI") -> None:
    st.markdown(
        f'<section class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{caption}</p></section>',
        unsafe_allow_html=True,
    )


def _metric(label: str, value: str, hint: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div><div class="hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _module_units(module_id: str):
    return [unit for unit in all_units() if unit.module_id == module_id]


def _status_badge(status: str) -> str:
    return f'<span class="badge {STATUS_CLASS[status]}">{STATUS_ICON[status]} {status}</span>'


def _report(assistant: LocalStudyAssistant) -> LearningReport:
    # Density scoring loads the reranker; one report per script run is enough.
    if "_report" not in st.session_state.get("_run_cache", {}):
        st.session_state["_run_cache"] = {"_report": assistant.learning_report()}
    return st.session_state["_run_cache"]["_report"]


def reset_run_cache() -> None:
    st.session_state["_run_cache"] = {}


def render_exam_date(assistant: LocalStudyAssistant) -> None:
    saved = assistant.progress.exam_date()
    chosen = st.date_input("시험 예정일", value=saved or date.today() + timedelta(days=30), min_value=date.today())
    if chosen != saved:
        assistant.progress.set_exam_date(chosen)
        reset_run_cache()


def _go(page: str, **state) -> None:
    """Button callback: switch page before the sidebar radio is drawn."""
    st.session_state["page"] = page
    st.session_state.update(state)


def render_dashboard(assistant: LocalStudyAssistant) -> None:
    _hero("수험 범위를 학습 지도로 바꾸다", "학습 메모·질문·퀴즈를 NCS 지식 지도에 투사해 실제로 확보한 지식량을 측정합니다.", "NOVA STUDY MAP · AI PLATFORM")
    report = _report(assistant)
    files = Counter(chunk.source for chunk in assistant.retriever.chunks)
    days = report.days_to_exam

    columns = st.columns(4)
    with columns[0]:
        reached = sum(item.reached for item in report.units)
        _metric("지식 커버리지", _percent(report.coverage), f"도달 {reached} / {len(report.units)} 단원 · 숙련 {report.mastery_coverage:.0%}")
    with columns[1]:
        _metric("의미적 학습 밀도", _percent(report.average_density), "메모와 NCS 원문의 의미 일치도")
    with columns[2]:
        _metric("시험일 유효 지식량", _percent(report.effective_knowledge),
                f"D-{days} 기준 망각 반영" if days is not None else "시험일을 설정하세요")
    with columns[3]:
        _metric("지식 연결성", _percent(report.connectivity), "선수 모듈이 준비된 학습 단원 비율")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="section-kicker">Learning coverage</div>', unsafe_allow_html=True)
        st.subheader("시험 범위별 숙련도")
        mastery = report.module_mastery()
        by_unit = report.by_unit()
        for module in MODULES:
            units = _module_units(module.module_id)
            reached_units = sum(by_unit[unit.unit_id].reached for unit in units)
            value = mastery.get(module.module_id, 0.0)
            st.markdown(f'<div class="module-card"><div class="module-name">{module.name}</div><div class="module-meta">숙련도 {value:.0%} · 도달 {reached_units} / {len(units)} 세부 단원</div></div>', unsafe_allow_html=True)
            st.progress(min(1.0, value))
    with right:
        st.markdown('<div class="section-kicker">AI pacemaker</div>', unsafe_allow_html=True)
        st.subheader("AI 학습 페이스메이커")
        if days is not None:
            pace = report.daily_pace
            st.caption(f"시험까지 **{days}일** · 남은 실질 학습량 **{report.remaining_units:.1f}단원**"
                       + (f" · 하루 권장 **{pace:.1f}단원**" if pace is not None else ""))
        if not report.recommendations:
            st.markdown('<div class="empty-card">학습 기록을 남기면 부족한 개념을 찾아 다음 행동을 제안합니다.</div>', unsafe_allow_html=True)
        for index, item in enumerate(report.recommendations):
            st.markdown(f'<div class="rec-card"><div class="rec-title">{item.title} · {item.unit.name}</div>'
                        f'<div class="rec-reason">{module_name(item.unit.module_id)} — {item.reason}</div></div>',
                        unsafe_allow_html=True)
            if item.action == "quiz":
                st.button("이 단원 퀴즈 풀기", key=f"rec-{index}", on_click=_go,
                          args=("확인 퀴즈",), kwargs={"pending_quiz_unit": item.unit.unit_id})
            elif item.action == "summary":
                if st.button("AI 요약 보기", key=f"rec-{index}"):
                    with st.spinner("NCS 근거로 요약을 만드는 중입니다..."):
                        st.session_state["agent_summary"] = (item.unit, *assistant.summarize_unit(item.unit))
            else:
                st.button("학습 기록 남기기", key=f"rec-{index}", on_click=_go, args=("학습 기록",))
        summary = st.session_state.get("agent_summary")
        if summary:
            unit, text, sources = summary
            with st.expander(f"AI 요약 · {unit.name}", expanded=True):
                st.markdown(text)
                _show_sources(sources)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">System status</div>', unsafe_allow_html=True)
        if assistant.index_ready:
            st.success(f"검색 인덱스 준비됨 · 근거 자료 {len(files)}개 · 검색 단위 {assistant.chunk_count}개")
        else:
            st.info("사이드바에서 자료 인덱스를 생성하면 AI 기능과 의미 분석을 사용할 수 있습니다.")


def render_study_log(assistant: LocalStudyAssistant) -> None:
    _hero("오늘의 학습을 남기세요", "메모는 NCS 원문과 비교해 학습 밀도를, 기록 날짜는 망각 곡선으로 복습 시점을 계산합니다.", "STUDY LOG")
    units = all_units()
    choices = {f"{module_name(unit.module_id)} · {unit.name}": unit for unit in units}
    write_tab, history_tab = st.tabs(["학습 기록 추가", "학습 히스토리"])
    with write_tab:
        with st.form("study-log", clear_on_submit=True):
            selected_label = st.selectbox("학습한 세부 단원", list(choices))
            col1, col2 = st.columns(2)
            with col1:
                status = st.selectbox("학습 상태", ["완료", "학습 중", "복습 필요"])
            with col2:
                minutes = st.number_input("학습 시간(분)", min_value=0, max_value=1440, value=30, step=5)
            note = st.text_area("학습 메모 (자신의 말로 요약할수록 학습 밀도가 정확해집니다)", placeholder="예: 요구사항 우선순위 도출 기준과 MoSCoW 방법을 정리했다.", height=130)
            submitted = st.form_submit_button("학습 기록 저장", type="primary", width="stretch")
        if submitted:
            unit = choices[selected_label]
            assistant.progress.save(unit.module_id, unit.unit_id, unit.name, status, int(minutes), note.strip())
            reset_run_cache()
            st.success("저장했습니다. 대시보드와 지식맵에 즉시 반영됩니다.")
            if note.strip() and assistant.unit_index.ready:
                density = assistant.unit_index.densities({unit.unit_id: [note.strip()]}).get(unit.unit_id)
                if density is not None:
                    st.info(f"이 메모와 NCS 원문의 의미 일치도: **{density:.0%}**")
    with history_tab:
        logs = assistant.progress.recent_logs(30)
        if not logs:
            st.markdown('<div class="empty-card">아직 학습 기록이 없습니다. 첫 단원을 기록해보세요.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(
                [{"시간": row[0], "단원": row[1], "상태": row[2], "학습시간(분)": row[3], "메모": row[4]} for row in logs],
                width="stretch",
                hide_index=True,
            )


def _node_color(item: UnitAnalytics) -> str:
    if not item.reached:
        return "#f2f4f7"
    value = item.mastery * item.retention_now
    return "#fee4e2" if value < 0.3 else "#fef0c7" if value < 0.55 else "#d1fadf" if value < 0.8 else "#75e0a7"


def knowledge_map_dot(report: LearningReport) -> str:
    """NCS modules as clusters, units as nodes coloured by retained mastery."""
    by_unit = report.by_unit()
    lines = [
        "digraph G {",
        'rankdir=LR; compound=true; nodesep=0.12; ranksep=0.55; bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fontname="Malgun Gothic", fontsize=10, color="#d0d5dd", penwidth=1];',
        'edge [color="#98a2b3", arrowsize=0.7];',
    ]
    for module in MODULES:
        units = _module_units(module.module_id)
        lines.append(f'subgraph cluster_{module.module_id} {{ label="{module.name}"; fontname="Malgun Gothic"; '
                     'fontsize=11; style="rounded"; color="#b2ccff";')
        for unit in units:
            item = by_unit[unit.unit_id]
            border = ', color="#f79009", penwidth=2' if item.status == "복습 필요" else ""
            lines.append(f'{unit.unit_id} [label="{unit.name}", fillcolor="{_node_color(item)}"{border}];')
        lines.append("{rank=same; " + "; ".join(unit.unit_id for unit in units) + "}")
        lines.append("}")
    for module in MODULES:
        target = _module_units(module.module_id)[0].unit_id
        for required in module.prerequisites:
            source = _module_units(required)[0].unit_id
            lines.append(f"{source} -> {target} [ltail=cluster_{required}, lhead=cluster_{module.module_id}];")
    lines.append("}")
    return "\n".join(lines)


def render_knowledge_map(assistant: LocalStudyAssistant) -> None:
    _hero("전체 수험 범위를 한눈에", "NCS 인공지능 플랫폼 구축 9개 모듈을 표준 지식 그래프로 삼아 학습 흔적과 기억 유지 상태를 투사합니다.", "KNOWLEDGE MAP")
    report = _report(assistant)
    counts = Counter(item.status for item in report.units)
    st.markdown(
        f"{_status_badge('완료')} {counts['완료']} &nbsp;&nbsp; {_status_badge('학습 중')} {counts['학습 중']} &nbsp;&nbsp; "
        f"{_status_badge('복습 필요')} {counts['복습 필요']} &nbsp;&nbsp; 색상: 회색 미도달 → 빨강 약함 → 노랑 → 초록 숙련(현재 기억 유지 반영)",
        unsafe_allow_html=True,
    )
    st.graphviz_chart(knowledge_map_dot(report), width="stretch")
    st.markdown('<div class="section-kicker">Unit analytics</div>', unsafe_allow_html=True)
    st.dataframe(
        [
            {
                "모듈": module_name(item.unit.module_id),
                "단원": item.unit.name,
                "상태": item.status,
                "숙련도": round(item.mastery * 100),
                "학습 밀도": None if item.density is None else round(item.density * 100),
                "퀴즈 정답률": None if item.quiz_accuracy is None else round(item.quiz_accuracy * 100),
                "질문 수": item.question_count,
                "복습 일수": item.review_days,
                "현재 기억 유지율": round(item.retention_now * 100),
                "시험일 기억 유지율": round(item.retention_exam * 100),
                "선수 학습 준비": "예" if item.prerequisites_ready else "아니오",
            }
            for item in report.units
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption("숙련도 = 학습 상태·학습 밀도·퀴즈 정답률의 가중 평균. 기억 유지율 = 0.25 + 0.75·exp(-경과일/안정도), "
               "안정도는 복습한 날이 늘수록 두 배씩, 퀴즈 정답률이 높을수록 커집니다.")


def _show_sources(sources: list[tuple]) -> None:
    if not sources:
        return
    st.markdown('<div class="section-kicker">Retrieved evidence</div>', unsafe_allow_html=True)
    for index, (chunk, score) in enumerate(sources, start=1):
        st.markdown(f'<div class="source-card"><b>근거 {index}</b> · {chunk.source}<br><span style="color:#667085;font-size:.78rem">질문 관련도 {reranker_probability(score):.1%} (cross-encoder)</span><br><br>{chunk.text}</div>', unsafe_allow_html=True)


def _show_stepwise_trace(assistant: LocalStudyAssistant) -> None:
    result = assistant.last_stepwise
    if result is None:
        return
    st.markdown('<div class="section-kicker">Stepwise retrieval control</div>', unsafe_allow_html=True)
    st.caption(f"논문 정책 `{result.policy_name}` · 총 검색 **{result.total_searches}회** (최대 4회) · {result.seconds:.1f}초. "
               "매 단계에서 후속 검색의 답변 F1 이득을 예측해 단계별 임곗값보다 크면 SEARCH, 아니면 STOP합니다.")
    for step in result.steps:
        if step.decision == "LIMIT":
            badge = '<span class="badge stop">최대 검색 도달</span>'
            signal = ""
        else:
            css = "search" if step.decision == "SEARCH" else "stop"
            badge = f'<span class="badge {css}">{step.decision}</span>'
            signal = f" · 예측 이득 {step.predicted_gain:+.4f} vs 임곗값 {step.threshold:+.4f}"
        query = "원 질문으로 최초 검색" if step.total_searches == 1 else f"갱신 질의: {step.query}"
        added = f" · 새 근거 {len(step.added)}개" if step.total_searches > 1 else ""
        st.markdown(
            f'<div class="step-card"><div class="step-head">검색 {step.total_searches}회 {badge}</div>'
            f'<div class="step-meta">{query}{added}{signal} · {step.seconds:.1f}초</div>'
            f'<div class="step-meta">중간 답변: <b>{step.prediction or "—"}</b></div></div>',
            unsafe_allow_html=True,
        )


def render_chat(assistant: LocalStudyAssistant) -> None:
    _hero("근거를 찾고, 필요한 만큼만 더 찾습니다", "논문의 답변 이득 예측 정책이 검색을 더 할지 멈출지 단계마다 판단한 뒤, 확보한 근거 안에서만 답합니다.", "AI STUDY ASSISTANT")
    if not assistant.ready_for_retrieval:
        st.markdown('<div class="empty-card"><b>검색 인덱스가 없거나 자료가 변경되었습니다.</b><br>왼쪽 사이드바에서 인덱스를 생성·갱신한 뒤 이용해주세요.</div>', unsafe_allow_html=True)
        return
    st.caption("추천 질문")
    suggestions = ["인공지능 플랫폼 요구사항 명세화의 핵심은 무엇이야?", "지식 베이스를 구축하는 절차를 설명해줘", "플랫폼 성능 테스트 항목을 정리해줘"]
    suggestion_columns = st.columns(3)
    for index, suggestion in enumerate(suggestions):
        with suggestion_columns[index]:
            if st.button(suggestion, key=f"suggestion-{index}", width="stretch"):
                st.session_state["question"] = suggestion
    question = st.text_area("궁금한 내용을 입력하세요", key="question", placeholder="예: 인공지능 플랫폼 요구사항을 어떻게 우선순위화해야 해?", height=115)
    if st.button("근거 기반 답변 생성", type="primary"):
        if not question.strip():
            st.warning("질문을 입력해주세요.")
            return
        with st.spinner("단계별로 근거를 찾고 검색 지속 여부를 판단하고 있습니다..."):
            answer, sources = assistant.answer(question)
        reset_run_cache()
        st.markdown("### AI 답변")
        st.markdown(answer)
        _show_stepwise_trace(assistant)
        _show_sources(sources)


def _unit_choices() -> dict[str, LearningUnit]:
    return {f"{module_name(unit.module_id)} · {unit.name}": unit for unit in all_units()}


def render_quiz(assistant: LocalStudyAssistant) -> None:
    _hero("약점을 확인하는 짧은 퀴즈", "단원의 NCS 원문 근거 안에서만 문제를 만들고, 채점 결과는 숙련도와 망각 예측에 반영됩니다.", "KNOWLEDGE CHECK")
    if not assistant.ready_for_retrieval:
        st.markdown('<div class="empty-card"><b>검색 인덱스가 없거나 자료가 변경되었습니다.</b><br>왼쪽 사이드바에서 인덱스를 생성·갱신한 뒤 이용해주세요.</div>', unsafe_allow_html=True)
        return
    choices = _unit_choices()
    labels = list(choices)
    pending = st.session_state.pop("pending_quiz_unit", None)
    default = next((index for index, label in enumerate(labels) if choices[label].unit_id == pending), 0)
    mode = st.radio("출제 기준", ["단원 선택", "자유 주제"], horizontal=True)
    if mode == "단원 선택":
        unit = choices[st.selectbox("복습할 단원", labels, index=default)]
        topic = unit.name
    else:
        unit = None
        topic = st.text_input("복습할 개념", placeholder="예: 지식 베이스 생성")
    if st.button("근거 기반 퀴즈 생성", type="primary") or pending:
        if not topic.strip():
            st.warning("복습할 개념을 입력해주세요.")
            return
        with st.spinner("근거 문서를 검토하고 퀴즈를 만드는 중입니다..."):
            quiz, sources = assistant.make_quiz(topic, unit)
        st.session_state["quiz"] = (quiz, sources)
        st.session_state.pop("quiz_graded", None)
    saved = st.session_state.get("quiz")
    if not saved:
        return
    quiz, sources = saved
    _render_quiz_form(assistant, quiz)
    _show_sources(sources)


def _render_quiz_form(assistant: LocalStudyAssistant, quiz: Quiz) -> None:
    st.markdown(f"### 확인 퀴즈 · {quiz.topic}")
    if not quiz.items:
        st.warning(quiz.message)
        return
    graded = st.session_state.get("quiz_graded")
    with st.form("quiz-form"):
        picks = [
            st.radio(f"{number}. {item.question}", item.options, index=None, key=f"quiz-{id(quiz)}-{number}")
            for number, item in enumerate(quiz.items, start=1)
        ]
        submitted = st.form_submit_button("채점하기", type="primary", disabled=graded is not None)
    if submitted and graded is None:
        correct = sum(pick == item.options[item.answer_index] for pick, item in zip(picks, quiz.items, strict=True))
        assistant.record_quiz(quiz, correct)
        reset_run_cache()
        graded = st.session_state["quiz_graded"] = (correct, tuple(picks))
    if graded is not None:
        correct, picks = graded
        st.success(f"{len(quiz.items)}문항 중 {correct}문항 정답 · 숙련도와 복습 일정에 반영했습니다.")
        for number, (pick, item) in enumerate(zip(picks, quiz.items, strict=True), start=1):
            answer = item.options[item.answer_index]
            mark = "✅" if pick == answer else "❌"
            st.markdown(f"{mark} **{number}. 정답: {answer}** — {item.explanation} [근거 {item.evidence}]")
