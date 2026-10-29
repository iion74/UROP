"""Figures for the KCI review: lambda trade-off and generation-cost trade-off."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_kci_results import (  # noqa: E402
    ANALYSIS, COLORS, DATASETS, DISPLAY_NAMES, FONT_PATH, configure_style, save_figure,
)

REPORT_PATH = ANALYSIS / "reviewer_requests.json"
WINDOWS_FONT = Path("C:/Windows/Fonts/malgun.ttf")
LABELED_LAMBDAS = ("0.001", "0.005", "0.01", "0.02", "0.05")


def _style() -> None:
    configure_style()
    if not FONT_PATH.exists() and WINDOWS_FONT.exists():
        font_manager.fontManager.addfont(str(WINDOWS_FONT))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(WINDOWS_FONT)).get_name()


def _mean(node: dict) -> float:
    return node["mean"]


def plot_lambda_tradeoff(report: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    sweep = report["across_seeds"]["lambda_sweep"]
    for axis, dataset, name, color in zip(axes, DATASETS, DISPLAY_NAMES, COLORS):
        fixed = report["reference"][dataset]["fixed"]
        axis.plot([1, 2, 3, 4], [fixed[k]["f1"] for k in "1234"], "s--", color="#777777",
                  markersize=3.5, linewidth=0.9, label="고정 1~4회")
        searches = [_mean(sweep[str(cost)][dataset]["average_total_searches"]) for cost in report["lambdas"]]
        f1 = [_mean(sweep[str(cost)][dataset]["f1"]) for cost in report["lambdas"]]
        axis.plot(searches, f1, "o-", color=color, markersize=3.2, linewidth=1.1, label="제안 정책(λ 변화)")
        for cost in LABELED_LAMBDAS:
            point = sweep[cost][dataset]
            axis.annotate(cost, (_mean(point["average_total_searches"]), _mean(point["f1"])),
                          textcoords="offset points", xytext=(2, -8), fontsize=6, color=color)
        axis.set_title(name)
        axis.set_xlabel("평균 검색 횟수")
        axis.set_xlim(0.9, 4.1)
        axis.grid(color="#DDDDDD", linewidth=0.55)
    axes[0].set_ylabel("F1")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig4_3_lambda_tradeoff")


def plot_cost_tradeoff(report: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    seeds = report["across_seeds"]
    for axis, dataset, name, color in zip(axes, DATASETS, DISPLAY_NAMES, COLORS):
        fixed = report["reference"][dataset]["fixed"]
        axis.plot([fixed[k]["estimated_latency_seconds"] for k in "1234"], [fixed[k]["f1"] for k in "1234"],
                  "s--", color="#777777", markersize=3.5, linewidth=0.9, label="고정 1~4회")
        for key, marker, label, alpha in (("lambda_sweep", "o-", "제안 정책(λ 변화)", 1.0),
                                          ("hybrid_reader_gate", "^:", "답변 생략 게이트", 0.75)):
            points = seeds[key].values()
            axis.plot([_mean(p[dataset]["estimated_latency_seconds"]) for p in points],
                      [_mean(p[dataset]["f1"]) for p in seeds[key].values()],
                      marker, color=color, alpha=alpha, markersize=3.0, linewidth=1.0, label=label)
        axis.set_title(name)
        axis.set_xlabel("추정 지연시간(초)")
        axis.grid(color="#DDDDDD", linewidth=0.55)
    axes[0].set_ylabel("F1")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig4_4_cost_tradeoff")


def main() -> None:
    _style()
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    plot_lambda_tradeoff(report)
    plot_cost_tradeoff(report)
    print("Saved fig4_3_lambda_tradeoff and fig4_4_cost_tradeoff")


if __name__ == "__main__":
    main()
