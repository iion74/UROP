"""System overview of the NOVA Study Map web app for the UROP final report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_kci_method_overview import (  # noqa: E402
    BLUE, BLUE_EDGE, DOUBLE_COLUMN_WIDTH, FONT_PATH, GRAY, GRAY_EDGE, GREEN, GREEN_EDGE,
    INK, ORANGE, ORANGE_EDGE, OUTPUT_DIR, arrow, box, configure_style,
)

WINDOWS_FONT = Path("C:/Windows/Fonts/NotoSansKR-Regular.ttf")
FRAME = "#F8F9FA"
FRAME_EDGE = "#ADB5BD"


def _style() -> None:
    configure_style()
    if not FONT_PATH.exists() and WINDOWS_FONT.exists():
        font_manager.fontManager.addfont(str(WINDOWS_FONT))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(WINDOWS_FONT)).get_name()


def frame(axis: plt.Axes, x: float, y: float, width: float, height: float, title: str,
          color: str, *, dashed: bool = False) -> None:
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.004,rounding_size=0.015",
        linewidth=0.9, edgecolor=FRAME_EDGE if dashed else color, facecolor="white" if dashed else FRAME,
        linestyle="--" if dashed else "-", zorder=0,
    ))
    axis.text(x + 0.012, y + height - 0.035, title, ha="left", va="center", fontsize=7.6,
              weight="bold", color=color, zorder=3)


def two_way(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], label: str,
            offset: tuple[float, float]) -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="<|-|>", mutation_scale=8, linewidth=1.0,
                                   color="#4B5563", shrinkA=1, shrinkB=1, zorder=1))
    axis.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], label,
              ha="center", va="center", fontsize=6.0, color="#4B5563", linespacing=1.15,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6}, zorder=4)


def main() -> None:
    _style()
    fig, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 100.0 / 25.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    frame(axis, 0.150, 0.015, 0.840, 0.970, "로컬 GPU 서버 (외부 통신 없음)", GRAY_EDGE, dashed=True)
    box(axis, 0.010, 0.775, 0.105, 0.115, "수험생", face=BLUE, edge=BLUE_EDGE, weight="bold")
    box(axis, 0.175, 0.775, 0.795, 0.115,
        "Streamlit 웹 화면\n대시보드 · 학습 기록 · 지식 맵 · AI 학습 도우미 · 확인 퀴즈",
        face=ORANGE, edge=ORANGE_EDGE, weight="bold")
    two_way(axis, (0.115, 0.832), (0.175, 0.832), "", (0, 0))

    # (1) Source processing
    frame(axis, 0.175, 0.105, 0.270, 0.585, "① 자료 처리", GRAY_EDGE)
    box(axis, 0.195, 0.510, 0.230, 0.095, "NCS 학습모듈 PDF (9권)", face=GRAY, edge=GRAY_EDGE)
    box(axis, 0.195, 0.345, 0.230, 0.110, "텍스트 정제 · 단원 배정\n794개 구절 → 63개 단원",
        face=GRAY, edge=GRAY_EDGE)
    box(axis, 0.195, 0.135, 0.103, 0.150, "표준 지식 지도\nNCS 단원 +\n모듈 선수관계", fontsize=6.6)
    box(axis, 0.322, 0.135, 0.103, 0.150, "하이브리드 색인\nBM25 · 벡터\n+ 재정렬", fontsize=6.6)
    arrow(axis, (0.310, 0.510), (0.310, 0.455))
    arrow(axis, (0.2465, 0.345), (0.2465, 0.285))
    arrow(axis, (0.3735, 0.345), (0.3735, 0.285))

    # (2) AI features
    frame(axis, 0.470, 0.105, 0.245, 0.585, "② AI 기능", GREEN_EDGE)
    box(axis, 0.488, 0.500, 0.209, 0.110, "AI 학습 도우미\n단계별 검색 제어 (논문)",
        face=GREEN, edge=GREEN_EDGE, weight="bold", fontsize=6.8)
    box(axis, 0.488, 0.385, 0.209, 0.080, "확인 퀴즈 · 단원 요약", face=GREEN, edge=GREEN_EDGE, fontsize=6.8)
    box(axis, 0.488, 0.255, 0.209, 0.095, "AI 페이스메이커\n취약 단원 탐지·보충 제안",
        face=GREEN, edge=GREEN_EDGE, fontsize=6.8)
    box(axis, 0.488, 0.130, 0.209, 0.090, "Qwen3-8B (Ollama)\n로컬 생성 모델", face=GRAY, edge=GRAY_EDGE,
        fontsize=6.8)
    arrow(axis, (0.5925, 0.220), (0.5925, 0.255))

    # (3) Learning analytics
    frame(axis, 0.740, 0.105, 0.230, 0.585, "③ 학습량 분석", BLUE_EDGE)
    box(axis, 0.758, 0.500, 0.194, 0.110, "학습 기록 DB (SQLite)\n메모 · 질문 · 퀴즈 결과", fontsize=6.6)
    box(axis, 0.758, 0.130, 0.194, 0.320,
        "지식 커버리지\n\n의미적 학습 밀도\n\n단원별 숙련도\n\n시험일 유효 지식량\n(망각 곡선)", fontsize=6.8)
    arrow(axis, (0.855, 0.500), (0.855, 0.450))

    # Data flow between the three parts and the screen
    arrow(axis, (0.433, 0.250), (0.488, 0.530))
    axis.text(0.4575, 0.375, "근거\n검색", ha="center", va="center", fontsize=6.0, color="#4B5563",
              linespacing=1.1, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5}, zorder=4)
    two_way(axis, (0.697, 0.300), (0.758, 0.300), "취약\n단원", (0.0, 0.0))
    two_way(axis, (0.697, 0.425), (0.758, 0.540), "퀴즈\n결과", (0.0, 0.0))
    arrow(axis, (0.2465, 0.127), (0.855, 0.121), connectionstyle="arc3,rad=0.14",
          label="질문·기록의 단원 매핑", label_offset=(0.0, -0.075))
    arrow(axis, (0.310, 0.775), (0.310, 0.690), label="자료 읽기", label_offset=(0.0, 0.0))
    two_way(axis, (0.5925, 0.690), (0.5925, 0.775), "", (0, 0))
    two_way(axis, (0.855, 0.690), (0.855, 0.775), "", (0, 0))

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "fig_urop_system_overview"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.[png|pdf|svg]")


if __name__ == "__main__":
    main()
