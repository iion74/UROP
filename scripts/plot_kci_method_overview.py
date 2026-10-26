"""Create a publication-ready overview of policy training and deployment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper/figures"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
DOUBLE_COLUMN_WIDTH = 174.0 / 25.4

BLUE = "#DCEEFE"
BLUE_EDGE = "#2979B8"
GREEN = "#DDF3E8"
GREEN_EDGE = "#198754"
ORANGE = "#FFF0D6"
ORANGE_EDGE = "#D97706"
GRAY = "#EFF1F3"
GRAY_EDGE = "#6C757D"
RED = "#FDE5E5"
RED_EDGE = "#C43D3D"
INK = "#20252B"


def configure_style() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(FONT_PATH)
        ).get_name()
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    face: str = BLUE,
    edge: str = BLUE_EDGE,
    fontsize: float = 7.2,
    dashed: bool = False,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.05,
        edgecolor=edge,
        facecolor=face,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        weight=weight,
        linespacing=1.22,
        zorder=3,
    )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#4B5563",
    dashed: bool = False,
    connectionstyle: str = "arc3",
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.0,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=connectionstyle,
        shrinkA=1,
        shrinkB=1,
        zorder=1,
    )
    axis.add_patch(patch)
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=6.3,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
            zorder=4,
        )


def decision_diamond(
    axis: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    text: str,
) -> None:
    x, y = center
    patch = Polygon(
        [(x, y + height / 2), (x + width / 2, y),
         (x, y - height / 2), (x - width / 2, y)],
        closed=True,
        facecolor=ORANGE,
        edgecolor=ORANGE_EDGE,
        linewidth=1.05,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=7.0,
              color=INK, linespacing=1.1, zorder=3)


def draw_training(axis: plt.Axes) -> None:
    axis.text(0.025, 0.93, "(a) 정책 학습", fontsize=9.2, weight="bold", color=INK)

    y, h = 0.65, 0.15
    boxes = [
        (0.035, 0.115, "Train 질의\n데이터셋별 2,000개"),
        (0.185, 0.135, "검색·답변 궤적\n1~4회"),
        (0.355, 0.125, "상태 특징\n$\\mathbf{s}_{i,t}$ (105차원)"),
        (0.515, 0.135, "미래 답변 이득\n$g_{i,t}$"),
        (0.685, 0.135, "그래디언트 부스팅\n회귀모델"),
        (0.855, 0.115, "학습된 정책\n$f_{\\theta},\\;\\tau_1{\\sim}\\tau_3$"),
    ]
    for index, (x, w, text) in enumerate(boxes):
        if index == 1:
            box(axis, x, y, w, h, text, face=GRAY, edge=GRAY_EDGE)
        elif index in (4, 5):
            box(axis, x, y, w, h, text, face=GREEN, edge=GREEN_EDGE,
                weight="bold" if index == 5 else "normal")
        else:
            box(axis, x, y, w, h, text)
        if index:
            previous_x, previous_w, _ = boxes[index - 1]
            arrow(axis, (previous_x + previous_w, y + h / 2), (x, y + h / 2))

    box(
        axis,
        0.39,
        0.48,
        0.12,
        0.10,
        "정답 $a_i^*$\n(학습에만 사용)",
        face=RED,
        edge=RED_EDGE,
        dashed=True,
        fontsize=6.8,
    )
    arrow(
        axis,
        (0.51, 0.53),
        (0.58, 0.65),
        color=RED_EDGE,
        dashed=True,
        label="단계별 F1 계산",
        label_offset=(0.025, -0.01),
    )
    axis.text(
        0.747,
        0.59,
        "여러 결정트리의 예측을 순차 결합",
        ha="center",
        va="center",
        fontsize=6.3,
        color=GREEN_EDGE,
    )


def draw_inference(axis: plt.Axes) -> None:
    axis.text(0.025, 0.39, "(b) 단계별 검색 제어", fontsize=9.2, weight="bold", color=INK)

    y, h = 0.16, 0.14
    box(axis, 0.035, y, 0.095, h, "입력 질문\n$q_i$")
    box(axis, 0.165, y, 0.115, h, "최초 검색\n(필수)", face=GRAY, edge=GRAY_EDGE)
    box(axis, 0.315, y, 0.145, h, "근거 + 현재 답변\nQwen3:8B (고정)", face=GRAY, edge=GRAY_EDGE)
    box(axis, 0.495, y, 0.125, h, "현재 상태\n$\\mathbf{s}_{i,t}$")
    box(axis, 0.655, y, 0.125, h, "예상 답변 이득\n$\\hat{g}_{i,t}=f_\\theta(\\mathbf{s}_{i,t})$",
        face=GREEN, edge=GREEN_EDGE)
    decision_diamond(axis, (0.845, y + h / 2), 0.115, 0.18,
                     "$\\hat{g}_{i,t}>\\tau_t$ ?")

    positions = [(0.13, 0.165), (0.28, 0.315), (0.46, 0.495), (0.62, 0.655)]
    widths = [0.095, 0.115, 0.145, 0.125]
    for (start_x, end_x), width in zip(positions, widths, strict=True):
        arrow(axis, (start_x, y + h / 2), (end_x, y + h / 2))
    arrow(axis, (0.78, y + h / 2), (0.7875, y + h / 2))

    box(axis, 0.92, 0.275, 0.065, 0.095, "STOP\n최종 답변", face=ORANGE,
        edge=ORANGE_EDGE, fontsize=6.7, weight="bold")
    arrow(axis, (0.895, 0.255), (0.92, 0.305), color=ORANGE_EDGE)
    axis.text(0.913, 0.292, "아니오", ha="right", va="center", fontsize=6.3,
              color=ORANGE_EDGE, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})

    box(axis, 0.775, 0.015, 0.14, 0.09, "SEARCH\n질의 갱신·다음 검색", face=ORANGE,
        edge=ORANGE_EDGE, fontsize=6.5, weight="bold")
    arrow(axis, (0.845, 0.14), (0.845, 0.105), color=ORANGE_EDGE, label="예",
          label_offset=(0.026, 0.0))
    arrow(
        axis,
        (0.775, 0.06),
        (0.39, 0.155),
        color=ORANGE_EDGE,
        connectionstyle="arc3,rad=-0.23",
        label="다음 단계로 반복 (최대 4회)",
        label_offset=(-0.02, -0.02),
    )

    axis.text(
        0.04,
        0.035,
        "※ 실제 의사결정에는 정답과 미래 검색 결과를 사용하지 않음",
        ha="left",
        va="center",
        fontsize=6.2,
        color=RED_EDGE,
    )


def main() -> None:
    configure_style()
    fig, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 103.0 / 25.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    draw_training(axis)
    axis.plot([0.025, 0.985], [0.445, 0.445], color="#CED4DA", linewidth=0.8)
    draw_inference(axis)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.015, top=0.99)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "fig3_1_method_overview"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.[png|pdf|svg]")


if __name__ == "__main__":
    main()
