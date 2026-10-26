"""Local certification-study assistant.

Run with: streamlit run UROP.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.assistant import LocalStudyAssistant
from src.document_loader import supported_files
from src.ui import (
    PAGES,
    inject_styles,
    render_chat,
    render_dashboard,
    render_exam_date,
    render_knowledge_map,
    render_quiz,
    render_study_log,
    reset_run_cache,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "data" / "index"


@st.cache_resource(show_spinner=False)
def get_assistant() -> LocalStudyAssistant:
    return LocalStudyAssistant(data_dir=DATA_DIR, index_dir=INDEX_DIR)


def main() -> None:
    st.set_page_config(page_title="NOVA Study Map", page_icon="🧭", layout="wide")
    inject_styles()
    assistant = get_assistant()

    with st.sidebar:
        st.markdown("## NOVA\nStudy Map")
        st.caption("LOCAL AI · EXAM PREP")
        st.markdown("<br>", unsafe_allow_html=True)
        page = st.radio("메뉴", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        render_exam_date(assistant)
        st.divider()
        st.markdown("#### Knowledge base")
        st.caption("`data/` 아래의 PDF, TXT, MD, CSV를 로컬 검색 범위로 사용합니다.")
        files = supported_files(DATA_DIR)
        st.write(f"발견한 자료 **{len(files)}개**")
        status = assistant.index_status
        button_label = "자료 읽기 및 인덱스 생성" if status.state == "missing" else "자료 변경 반영 (전체 인덱스 갱신)"
        if st.button(button_label, type="primary", width="stretch"):
            with st.spinner("문서를 읽고 검색 인덱스를 만드는 중입니다..."):
                result = assistant.build_index()
            reset_run_cache()
            st.success(f"{result.document_count}개 문서 · {result.chunk_count}개 검색 단위")
        status = assistant.index_status
        if status.state == "ready":
            st.success(f"검색 준비됨 · {assistant.chunk_count}개 학습 단위")
            st.caption(status.message)
        elif status.state == "stale":
            st.warning(status.message)
        else:
            st.info("자료를 넣고 인덱스를 생성하면 AI 기능을 사용할 수 있습니다.")
        st.divider()
        st.markdown("#### Local model")
        st.caption(
            "Generation · `qwen3:8b`\n\n"
            "Answer check · structured evidence chain\n\n"
            "Retrieval · multilingual-e5-base + bge-reranker\n\n"
            f"Search control · {assistant.search_mode}"
            + (f" · `{assistant.policy_name}`" if assistant.policy_name else "")
        )

    if page == "대시보드":
        render_dashboard(assistant)
    elif page == "학습 기록":
        render_study_log(assistant)
    elif page == "지식 맵":
        render_knowledge_map(assistant)
    elif page == "AI 학습 도우미":
        render_chat(assistant)
    else:
        render_quiz(assistant)


if __name__ == "__main__":
    main()
