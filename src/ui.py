from __future__ import annotations

from collections import Counter

import streamlit as st

from .assistant import LocalStudyAssistant
from .curriculum import MODULES, all_units, module_name


STATUS_ICON = {"완료": "●", "복습 필요": "●", "학습 중": "●", "미학습": "○"}
STATUS_CLASS = {"완료": "done", "복습 필요": "review", "학습 중": "active", "미학습": "idle"}


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
        .source-card { background: #f8fafc; border-left: 3px solid #7f9cff; border-radius: 0 10px 10px 0; padding: .85rem 1rem; margin: .55rem 0; }
        .empty-card { background: white; border: 1px dashed #d0d5dd; border-radius: 16px; padding: 1.3rem; color: #667085; }
        [data-testid="stForm"] { border: 1px solid #e4e7ec; border-radius: 16px; background: white; padding: 1.1rem; }
        .stButton > button[kind="primary"] { background: #244b9b; border-color: #244b9b; border-radius: 9px; font-weight: 700; }
        .stButton > button[kind="primary"]:hover { background: #173875; border-color: #173875; }
        .stProgress > div > div { background: #3157a9; }
        </style>
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


def _module_units(module_id: str):
    return [unit for unit in all_units() if unit.module_id == module_id]


def _module_progress(module_id: str, statuses: dict[str, str]) -> tuple[int, int]:
    units = _module_units(module_id)
    return sum(statuses.get(unit.unit_id) == "완료" for unit in units), len(units)


def _status_badge(status: str) -> str:
    return f'<span class="badge {STATUS_CLASS[status]}">{STATUS_ICON[status]} {status}</span>'


def render_dashboard(assistant: LocalStudyAssistant) -> None:
    _hero("수험 범위를 학습 지도로 바꾸다", "공부한 단원을 기록하고, AI가 NCS 기반 범위에서 근거를 찾아 설명합니다.", "NOVA STUDY MAP · AI PLATFORM")
    summary = assistant.progress.summary(len(all_units()))
    statuses = assistant.progress.latest_statuses()
    files = Counter(chunk.source for chunk in assistant.retriever.chunks)

    columns = st.columns(4)
    with columns[0]:
        _metric("학습 커버리지", f"{summary.coverage:.0f}%", f"완료 {summary.completed_units} / {summary.total_units} 단원")
    with columns[1]:
        _metric("복습 필요", f"{summary.review_units}개", "기록된 학습 상태 기준")
    with columns[2]:
        _metric("누적 학습시간", f"{summary.total_minutes}분", "학습 기록 합계")
    with columns[3]:
        _metric("AI 근거 자료", f"{len(files)}개", f"검색 단위 {assistant.chunk_count}개")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="section-kicker">Learning coverage</div>', unsafe_allow_html=True)
        st.subheader("시험 범위별 진행 상황")
        for module in MODULES:
            completed, total = _module_progress(module.module_id, statuses)
            progress = completed / total
            st.markdown(f'<div class="module-card"><div class="module-name">{module.name}</div><div class="module-meta">완료 {completed} / {total} 세부 단원 · {progress:.0%}</div></div>', unsafe_allow_html=True)
            st.progress(progress)
    with right:
        st.markdown('<div class="section-kicker">Next action</div>', unsafe_allow_html=True)
        st.subheader("다음 학습 제안")
        pending = next((unit for unit in all_units() if statuses.get(unit.unit_id) != "완료"), None)
        if pending:
            state = statuses.get(pending.unit_id, "미학습")
            st.markdown('<div class="empty-card">', unsafe_allow_html=True)
            st.markdown(f"**{module_name(pending.module_id)}**")
            st.markdown(f"### {pending.name}")
            st.markdown(_status_badge(state), unsafe_allow_html=True)
            st.caption("학습 기록을 남기면 수험 범위 커버리지가 갱신됩니다.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.success("모든 세부 단원을 완료했습니다. 복습 필요 단원을 다시 확인해보세요.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">System status</div>', unsafe_allow_html=True)
        if assistant.index_ready:
            st.success("Hybrid RAG 검색 인덱스가 준비되었습니다.")
        else:
            st.info("사이드바에서 자료 인덱스를 생성하면 AI 학습 도우미를 사용할 수 있습니다.")


def render_study_log(assistant: LocalStudyAssistant) -> None:
    _hero("오늘의 학습을 남기세요", "세부 단원의 완료 상태, 학습 시간, 메모를 로컬에 저장해 커버리지와 복습 목록을 관리합니다.", "STUDY LOG")
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
            note = st.text_area("학습 메모", placeholder="예: 요구사항 우선순위 도출 기준과 MoSCoW 방법을 정리했다.", height=130)
            submitted = st.form_submit_button("학습 기록 저장", type="primary", use_container_width=True)
        if submitted:
            unit = choices[selected_label]
            assistant.progress.save(unit.module_id, unit.unit_id, unit.name, status, int(minutes), note.strip())
            st.success("저장했습니다. 대시보드와 지식맵에 즉시 반영됩니다.")
    with history_tab:
        logs = assistant.progress.recent_logs(30)
        if not logs:
            st.markdown('<div class="empty-card">아직 학습 기록이 없습니다. 첫 단원을 기록해보세요.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(
                [{"시간": row[0], "단원": row[1], "상태": row[2], "학습시간(분)": row[3], "메모": row[4]} for row in logs],
                use_container_width=True,
                hide_index=True,
            )


def render_knowledge_map(assistant: LocalStudyAssistant) -> None:
    _hero("전체 수험 범위를 한눈에", "NCS 인공지능 플랫폼 구축 모듈의 선수관계와 세부 학습 단원을 지도로 정리했습니다.", "KNOWLEDGE MAP")
    statuses = assistant.progress.latest_statuses()
    top_left, top_right = st.columns([1.5, 1])
    with top_left:
        st.subheader("학습 흐름")
        st.markdown("`구축 계획 → 요구사항 분석 → 플랫폼 설계 → 구현 4개 영역 → 테스트 → 품질 관리`")
    with top_right:
        counts = Counter(statuses.values())
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"{_status_badge('완료')} {counts['완료']} &nbsp;&nbsp; {_status_badge('학습 중')} {counts['학습 중']} &nbsp;&nbsp; {_status_badge('복습 필요')} {counts['복습 필요']}", unsafe_allow_html=True)

    for module in MODULES:
        completed, total = _module_progress(module.module_id, statuses)
        prerequisite = ", ".join(module_name(item) for item in module.prerequisites) or "선수 학습 없음"
        with st.expander(f"{module.name}  ·  완료 {completed}/{total}", expanded=module.module_id in {"plan", "design"}):
            st.caption(f"선수 학습: {prerequisite}")
            columns = st.columns(2)
            for index, unit in enumerate(_module_units(module.module_id)):
                status = statuses.get(unit.unit_id, "미학습")
                with columns[index % 2]:
                    st.markdown(f"{_status_badge(status)} &nbsp; **{unit.name}**", unsafe_allow_html=True)


def _show_sources(sources: list[tuple]) -> None:
    if not sources:
        return
    st.markdown('<div class="section-kicker">Retrieved evidence</div>', unsafe_allow_html=True)
    for index, (chunk, score) in enumerate(sources, start=1):
        st.markdown(f'<div class="source-card"><b>근거 {index}</b> · {chunk.source}<br><span style="color:#667085;font-size:.78rem">Cross-encoder relevance score {score:.1%}</span><br><br>{chunk.text}</div>', unsafe_allow_html=True)


def _show_retrieval_trace(assistant: LocalStudyAssistant) -> None:
    trace = assistant.retrieval_trace
    if trace is None:
        return
    method = "Hybrid Vector RAG" if trace.graph_hops == 0 else f"Evidence-Gain Graph RAG · {trace.graph_hops}-hop"
    reason = ", ".join(trace.reasons)
    confidence = "충분" if trace.evidence_sufficient else "부족 — 답변 보류"
    profile = trace.profile
    policy_signal = (
        f"학습 정책 신뢰도 {trace.policy_confidence:.1%}"
        if trace.policy_confidence is not None
        else f"예상 순효용 {trace.expected_utility:.1%}"
    )
    st.caption(
        f"검색 정책: **{method}** · 초기 {trace.initial_candidate_count}개 → 재정렬 {trace.reranked_candidate_count}개 "
        f"· 그래프 추가 후보 {trace.expansion_budget}개 · 예상 그래프 근거효용 {trace.expected_gain:.1%} "
        f"· {policy_signal} · 근거 신뢰도 {trace.evidence_score:.1%} ({confidence})"
    )
    st.caption(
        f"근거 프로파일 — 불확실성 {profile.uncertainty:.1%}, 출처 분산 {profile.source_dispersion:.1%}, "
        f"그래프 연결성 {profile.graph_connectivity:.1%}, 중복도 {profile.redundancy:.1%} · 판단 근거: `{reason}`"
    )


def render_chat(assistant: LocalStudyAssistant) -> None:
    _hero("근거를 찾고, 이해를 돕습니다", "BM25와 의미 검색을 결합한 Hybrid RAG로 등록된 수험 범위 안에서만 답합니다.", "AI STUDY ASSISTANT")
    if not assistant.ready_for_retrieval:
        st.markdown('<div class="empty-card"><b>검색 인덱스가 없거나 자료가 변경되었습니다.</b><br>왼쪽 사이드바에서 인덱스를 생성·갱신한 뒤 이용해주세요.</div>', unsafe_allow_html=True)
        return
    st.caption("추천 질문")
    suggestions = ["인공지능 플랫폼 요구사항 명세화의 핵심은 무엇이야?", "지식 베이스를 구축하는 절차를 설명해줘", "플랫폼 성능 테스트 항목을 정리해줘"]
    suggestion_columns = st.columns(3)
    for index, suggestion in enumerate(suggestions):
        with suggestion_columns[index]:
            if st.button(suggestion, key=f"suggestion-{index}", use_container_width=True):
                st.session_state["question"] = suggestion
    question = st.text_area("궁금한 내용을 입력하세요", key="question", placeholder="예: 인공지능 플랫폼 요구사항을 어떻게 우선순위화해야 해?", height=115)
    if st.button("근거 기반 답변 생성", type="primary"):
        if not question.strip():
            st.warning("질문을 입력해주세요.")
            return
        with st.spinner("학습 자료에서 관련 근거를 찾고 있습니다..."):
            answer, sources = assistant.answer(question)
        st.markdown("### AI 답변")
        st.markdown(answer)
        _show_retrieval_trace(assistant)
        _show_sources(sources)


def render_quiz(assistant: LocalStudyAssistant) -> None:
    _hero("약점을 확인하는 짧은 퀴즈", "선택한 개념과 관련된 NCS 문서를 먼저 찾고, 그 근거 안에서만 확인 문제를 생성합니다.", "KNOWLEDGE CHECK")
    if not assistant.ready_for_retrieval:
        st.markdown('<div class="empty-card"><b>검색 인덱스가 없거나 자료가 변경되었습니다.</b><br>왼쪽 사이드바에서 인덱스를 생성·갱신한 뒤 이용해주세요.</div>', unsafe_allow_html=True)
        return
    left, right = st.columns([1.6, 1])
    with left:
        topic = st.text_input("복습할 개념", placeholder="예: 지식 베이스 생성")
    with right:
        st.caption("생성 형식")
        st.markdown("**객관식 3문항 · 보기 4개 · 해설 포함**")
    if st.button("근거 기반 퀴즈 생성", type="primary"):
        if not topic.strip():
            st.warning("복습할 개념을 입력해주세요.")
            return
        with st.spinner("근거 문서를 검토하고 퀴즈를 만드는 중입니다..."):
            quiz, sources = assistant.make_quiz(topic)
        st.markdown("### 생성된 확인 퀴즈")
        st.markdown(quiz)
        _show_retrieval_trace(assistant)
        _show_sources(sources)
