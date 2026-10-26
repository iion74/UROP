"""Create simple one-column figures for the KCI paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data/benchmarks/multihop/context_reader_aware/analysis"
SUMMARY_PATH = ANALYSIS / "sequential_answer_gain_multiseed.json"
SEED_REPORT_DIR = ANALYSIS / "sequential_multiseed"
OUTPUT_DIR = ROOT / "paper/figures"

DATASETS = ("hotpotqa", "musique", "2wiki")
DISPLAY_NAMES = ("HotpotQA", "MuSiQue", "2Wiki")
SEEDS = (42, 43, 44, 45, 46)
COLORS = ("#0072B2", "#D55E00", "#009E73")
SINGLE_COLUMN_WIDTH = 84.0 / 25.4
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def configure_style() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(FONT_PATH)
        ).get_name()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_results() -> tuple[dict, list[dict]]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    reports = [
        json.loads((SEED_REPORT_DIR / f"policy_seed{seed}.json").read_text(encoding="utf-8"))
        for seed in SEEDS
    ]
    return summary, reports


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_axis(axis: plt.Axes) -> None:
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.7, pad=2)


def plot_main_result(summary: dict) -> None:
    delta_f1 = np.asarray(
        [summary["datasets"][dataset]["across_seeds"]["delta_f1_points"]["mean"]
         for dataset in DATASETS]
    )
    delta_f1_std = np.asarray(
        [summary["datasets"][dataset]["across_seeds"]["delta_f1_points"]["sample_std"]
         for dataset in DATASETS]
    )
    search_reduction = np.asarray(
        [summary["datasets"][dataset]["across_seeds"]["search_reduction_percent"]["mean"]
         for dataset in DATASETS]
    )
    search_reduction_std = np.asarray(
        [summary["datasets"][dataset]["across_seeds"]["search_reduction_percent"]["sample_std"]
         for dataset in DATASETS]
    )
    y = np.arange(len(DATASETS), dtype=float)[::-1]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(SINGLE_COLUMN_WIDTH, 88.0 / 25.4),
        constrained_layout=True,
    )

    axes[0].barh(y, delta_f1, color=COLORS, height=0.55)
    axes[0].errorbar(
        delta_f1,
        y,
        xerr=delta_f1_std,
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
    )
    axes[0].set_yticks(y, DISPLAY_NAMES)
    axes[0].set_xlim(0, 1.55)
    axes[0].set_xlabel("고정 4회 대비 F1 향상 (점)")
    axes[0].set_title("(a) 답변 성능", loc="left", pad=4)
    for position, mean, std in zip(y, delta_f1, delta_f1_std, strict=True):
        axes[0].text(
            mean + std + 0.04,
            position,
            f"+{mean:.2f}±{std:.2f}",
            va="center",
            fontsize=6.8,
        )
    format_axis(axes[0])

    axes[1].barh(y, search_reduction, color=COLORS, height=0.55)
    axes[1].errorbar(
        search_reduction,
        y,
        xerr=search_reduction_std,
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
    )
    axes[1].set_yticks(y, DISPLAY_NAMES)
    axes[1].set_xlim(0, 48)
    axes[1].set_xlabel("고정 4회 대비 검색 감소율 (%)")
    axes[1].set_title("(b) 검색 횟수", loc="left", pad=4)
    for position, mean, std in zip(y, search_reduction, search_reduction_std, strict=True):
        axes[1].text(
            mean + std + 1.0,
            position,
            f"{mean:.1f}±{std:.1f}%",
            va="center",
            fontsize=6.8,
        )
    format_axis(axes[1])
    save_figure(fig, "fig4_1_main_result")


def plot_allocation_effect(reports: list[dict]) -> None:
    gains = np.asarray(
        [
            [
                report["evaluations"]["full_validation"][dataset][
                    "f1_change_vs_matched_depth_random_points"
                ]
                for report in reports
            ]
            for dataset in DATASETS
        ],
        dtype=float,
    )
    means = gains.mean(axis=1)
    stds = gains.std(axis=1, ddof=1)
    y = np.arange(len(DATASETS), dtype=float)[::-1]
    fig, axis = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, 53.0 / 25.4), constrained_layout=True
    )
    axis.barh(y, means, color=COLORS, height=0.55)
    axis.errorbar(
        means,
        y,
        xerr=stds,
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
    )
    axis.set_yticks(y, DISPLAY_NAMES)
    axis.set_xlim(0, 2.8)
    axis.set_xlabel("무작위 종료 대비 F1 향상 (점)")
    for position, mean, std in zip(y, means, stds, strict=True):
        axis.text(
            mean + std + 0.06,
            position,
            f"+{mean:.2f}±{std:.2f}",
            va="center",
            fontsize=6.8,
        )
    format_axis(axis)
    save_figure(fig, "fig4_2_allocation_effect")


def main() -> None:
    configure_style()
    summary, reports = load_results()
    plot_main_result(summary)
    plot_allocation_effect(reports)
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
