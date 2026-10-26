"""Create two one-column method figures for the two-column KCI layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from plot_kci_method_overview import (
    BLUE,
    BLUE_EDGE,
    GRAY,
    GRAY_EDGE,
    GREEN,
    GREEN_EDGE,
    INK,
    ORANGE,
    ORANGE_EDGE,
    RED,
    RED_EDGE,
    arrow,
    box,
    configure_style,
    decision_diamond,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper/figures"
SINGLE_COLUMN_WIDTH = 84.0 / 25.4


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / name
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def system_flow() -> None:
    fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 108.0 / 25.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    x, width, height = 0.18, 0.64, 0.085
    rows = [
        (0.89, "입력 질문 $q_i$", BLUE, BLUE_EDGE),
        (0.75, "최초 검색 (필수)", GRAY, GRAY_EDGE),
        (0.61, "근거 기반 현재 답변\nQwen3-8B (고정)", GRAY, GRAY_EDGE),
        (0.47, "현재 상태 특징 $\\mathbf{s}_{i,t}$\n질문·검색·근거·답변 (105차원)", BLUE, BLUE_EDGE),
        (0.32, "예상 답변 이득\n$\\hat{g}_{i,t}=f_\\theta(\\mathbf{s}_{i,t})$", GREEN, GREEN_EDGE),
    ]
    for y, text, face, edge in rows:
        box(axis, x, y, width, height, text, face=face, edge=edge, fontsize=7.0)
    for first, second in zip(rows, rows[1:]):
        arrow(axis, (0.5, first[0]), (0.5, second[0] + height))

    decision_diamond(axis, (0.5, 0.19), 0.34, 0.115, "$\\hat{g}_{i,t}>\\tau_t$ ?")
    arrow(axis, (0.5, 0.32), (0.5, 0.2475))

    box(axis, 0.70, 0.13, 0.27, 0.085, "STOP\n현재 답변 반환", face=ORANGE,
        edge=ORANGE_EDGE, fontsize=6.7, weight="bold")
    arrow(axis, (0.67, 0.19), (0.70, 0.172), color=ORANGE_EDGE)
    axis.text(0.695, 0.215, "아니오", ha="center", va="center", fontsize=6.2,
              color=ORANGE_EDGE)

    box(axis, 0.35, 0.015, 0.30, 0.085, "SEARCH\n질의 갱신·추가 검색", face=ORANGE,
        edge=ORANGE_EDGE, fontsize=6.7, weight="bold")
    arrow(axis, (0.5, 0.1325), (0.5, 0.10), color=ORANGE_EDGE)
    axis.text(0.56, 0.116, "예", ha="center", va="center", fontsize=6.2,
              color=ORANGE_EDGE)
    arrow(
        axis,
        (0.35, 0.055),
        (0.18, 0.65),
        color=ORANGE_EDGE,
        connectionstyle="arc3,rad=-0.28",
        label="다음 단계 (최대 4회)",
        label_offset=(-0.08, -0.01),
    )

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    save(fig, "fig3_1_system_flow")


def policy_learning() -> None:
    fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 101.0 / 25.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    box(axis, 0.18, 0.88, 0.64, 0.08, "Train 질문\n데이터셋별 시드당 2,000개",
        face=BLUE, edge=BLUE_EDGE, fontsize=6.8)
    box(axis, 0.18, 0.74, 0.64, 0.08, "검색·답변 궤적\n질문별 1~4회 검색",
        face=GRAY, edge=GRAY_EDGE, fontsize=6.8)
    arrow(axis, (0.5, 0.88), (0.5, 0.82))

    box(axis, 0.03, 0.55, 0.42, 0.10, "상태 입력 $\\mathbf{s}_{i,t}$\n관측 가능한 105개 특징",
        face=BLUE, edge=BLUE_EDGE, fontsize=6.6)
    box(axis, 0.55, 0.55, 0.42, 0.10, "학습 목표 $g_{i,t}$\n향후 최대 F1 이득",
        face=BLUE, edge=BLUE_EDGE, fontsize=6.6)
    arrow(axis, (0.40, 0.74), (0.24, 0.65))
    arrow(axis, (0.60, 0.74), (0.76, 0.65))

    box(axis, 0.64, 0.41, 0.30, 0.075, "정답 $a_i^*$\n레이블 생성에만 사용",
        face=RED, edge=RED_EDGE, dashed=True, fontsize=6.1)
    arrow(axis, (0.79, 0.485), (0.79, 0.55), color=RED_EDGE, dashed=True)

    box(axis, 0.16, 0.27, 0.68, 0.115,
        "HistGradientBoostingRegressor\n결정트리 예측을 순차 결합 (최대 180회)",
        face=GREEN, edge=GREEN_EDGE, fontsize=6.8)
    arrow(axis, (0.24, 0.55), (0.39, 0.385))
    arrow(axis, (0.76, 0.55), (0.61, 0.385))

    box(axis, 0.15, 0.12, 0.70, 0.085,
        "질문 단위 3-fold OOF 임곗값 보정\n검색 이득 $-$ 추가 검색 비용 최대화",
        face=GRAY, edge=GRAY_EDGE, fontsize=6.3)
    arrow(axis, (0.5, 0.27), (0.5, 0.205))

    box(axis, 0.23, 0.02, 0.54, 0.065,
        "최종 정책 $f_\\theta$와 단계별 임곗값 $\\tau_1,\\tau_2,\\tau_3$",
        face=GREEN, edge=GREEN_EDGE, fontsize=6.2, weight="bold")
    arrow(axis, (0.5, 0.12), (0.5, 0.085))

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    save(fig, "fig3_2_policy_learning")


def main() -> None:
    configure_style()
    system_flow()
    policy_learning()
    print(f"Saved split method figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
