"""
Plots for the emotion-steering result files.

    python plots/plot_emotion_results.py            # writes PNGs next to this file

Reads (paths relative to qwen_testing/):
    conceptor_emotion/conceptor_evaluation_*.csv    conceptor steering, BART-MNLI confidence
    confidence_score/alpha=*.csv                    additive EV steering, BART-MNLI confidence
    EAS_score/EAS_alpha=*.csv                       additive EV steering, LLM-judged EAS

A response counts as degenerate when under 5% of its characters are spaces
(word salad such as "OhWowAhOhImagine..."). Classifier confidence on such text
is not evidence of emotional steering, so the conceptor figure reports it.
"""

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

EMOTIONS = ["joy", "anger", "disgust", "fear", "sadness"]
COLORS = {"joy": "#E8A33D", "anger": "#D1495B", "disgust": "#6A994E",
          "fear": "#5E548E", "sadness": "#2E86AB"}
BASE_C, STEER_C = "#A0A4AB", "#E8643A"

plt.rcParams.update({
    "figure.facecolor": "#FCFCFA", "axes.facecolor": "#FCFCFA",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "axes.axisbelow": True,
    "font.size": 11,
})


def is_degenerate(text):
    text = text if isinstance(text, str) else ""
    return text.count(" ") / max(len(text), 1) < 0.05


def titles(fig, title, subtitle):
    fig.text(0.06, 0.965, title, fontsize=16, fontweight="bold", va="top")
    fig.text(0.06, 0.905, subtitle, fontsize=11, color="#555555", va="top")


def plot_conceptor():
    path = sorted(glob.glob(str(ROOT / "conceptor_emotion/conceptor_evaluation_*[0-9].csv")))[-1]
    df = pd.read_csv(path)
    beta = df["alpha"].iloc[0]
    aperture = re.search(r"aperture([\d.]+)_", path).group(1)
    df["degenerate"] = df["enhanced_response"].map(is_degenerate)

    g = df.groupby("emotion").agg(base=("baseline_max_confidence", "mean"),
                                  steer=("enhanced_max_confidence", "mean"),
                                  win=("confidence_improvement", lambda s: (s > 0).mean()),
                                  degen=("degenerate", "mean")).reindex(EMOTIONS)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.08, right=0.97)
    x, w = np.arange(len(EMOTIONS)), 0.38
    ax.bar(x - w / 2, g["base"], w, color=BASE_C, label="Baseline (no steering)")
    bars = ax.bar(x + w / 2, g["steer"], w, color=STEER_C, label="Conceptor-steered")
    for b, win in zip(bars, g["win"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{win:.0%} improved", ha="center", fontsize=9.5, color="#333")
    ax.set_xticks(x, [e.capitalize() for e in EMOTIONS])
    ax.set_ylabel("Mean target-emotion confidence (BART-MNLI)")
    ax.set_ylim(0, max(g["steer"].max(), g["base"].max()) * 1.25)
    ax.legend(frameon=False, loc="upper right")
    titles(fig, "Conceptor emotion steering — baseline vs steered confidence",
           f"Qwen2.5-1.5B-Instruct · h' = h + βC(h − μ) at all layers · β={beta:g}, aperture={aperture} · "
           f"{len(df) // len(EMOTIONS)} queries/emotion")
    fig.text(0.06, 0.035,
             f"⚠ {df['degenerate'].mean():.0%} of steered responses are degenerate word salad "
             "(e.g. “OhWowAhOhImagine…”).\n   The confidence gain is not evidence of coherent "
             "emotional text.",
             fontsize=10, color="#B3261E")
    fig.savefig(OUT / "conceptor_emotion_confidence.png", dpi=200)
    plt.close(fig)


def plot_ev_alpha():
    steered, baselines = [], []
    for path in glob.glob(str(ROOT / "confidence_score/alpha=*.csv")):
        d = pd.read_csv(path).groupby("emotion")
        steered.append(d["enhanced_max_confidence"].mean().rename(float(path.split("=")[-1][:-4])))
        baselines.append(d["baseline_max_confidence"].mean())
    # alpha=0 is the unsteered baseline; each run re-generated it, so average them
    base = pd.concat(baselines, axis=1).mean(axis=1).rename(0.0)
    table = pd.concat(steered + [base], axis=1)
    table = table[sorted(table.columns)].reindex(EMOTIONS)
    alphas = list(table.columns)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.08, right=0.86)
    mean = table.mean()
    # end-of-line labels, pushed apart so close values do not overlap
    ends = {e: table.loc[e].iloc[-1] for e in EMOTIONS}
    ends["Mean"] = mean.iloc[-1]
    label_y, prev = {}, None
    for name, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y = y if prev is None else max(y, prev + 0.042)
        label_y[name], prev = y, y
    for e in EMOTIONS:
        ax.plot(alphas, table.loc[e], marker="o", lw=2, color=COLORS[e])
        ax.text(alphas[-1] + 0.08, label_y[e], e.capitalize(),
                color=COLORS[e], fontweight="bold", va="center")
    ax.plot(alphas, mean, color="#222", lw=2.5, ls="--", marker="s")
    ax.text(alphas[-1] + 0.08, label_y["Mean"], f"Mean {mean.iloc[-1]:.2f}",
            fontweight="bold", va="center")
    for a, m in zip(alphas[:-1], mean.iloc[:-1]):
        ax.annotate(f"mean {m:.2f}", (a, m), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#FCFCFA", ec="none"))
    ax.axvline(0, color="#999", lw=1, ls=":")
    ax.set_xticks(alphas, [f"{a:g}" + (" (baseline)" if a == 0 else "") for a in alphas])
    ax.set_xlabel("Steering strength α (unit-norm emotion vector, all layers)")
    ax.set_ylabel("Mean target-emotion confidence (BART-MNLI)")
    ax.set_ylim(0, 1)
    titles(fig, "Additive emotion vectors — confidence vs steering strength",
           "Qwen2.5-1.5B-Instruct · h' = h + α·v̂ at the last token · 50 queries/emotion · "
           "0% degenerate outputs at every α")
    fig.savefig(OUT / "ev_confidence_vs_alpha.png", dpi=200)
    plt.close(fig)


def plot_eas():
    runs = {}
    for path in glob.glob(str(ROOT / "EAS_score/EAS_alpha=*.csv")):
        d = pd.read_csv(path)
        runs[float(d["alpha"].iloc[0])] = d
    alphas = sorted(runs)
    judge_emotions = EMOTIONS + ["surprise"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.6]})
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.06, right=0.98, wspace=0.22)

    # left: absolute EAS, baseline vs steered, per alpha run
    x, w = np.arange(len(alphas)), 0.36
    base = [runs[a]["baseline_eas"].mean() for a in alphas]
    steer = [runs[a]["enhanced_eas"].mean() for a in alphas]
    ax1.bar(x - w / 2, base, w, color=BASE_C, label="Baseline")
    ax1.bar(x + w / 2, steer, w, color=STEER_C, label="EV-steered")
    for i, a in enumerate(alphas):
        diff = runs[a]["eas_difference"]
        ax1.text(i, max(base[i], steer[i]) + 0.04, f"Δ = {diff.mean():+.2f}\n(n={len(diff)})",
                 ha="center", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#FCFCFA", ec="none"))
    ax1.axhline(0.5, color="#999", lw=1, ls=":", zorder=0)
    ax1.text(-0.55, 0.52, "clear emotion (0.5)", fontsize=8.5, color="#777", ha="left")
    ax1.set_xticks(x, [f"α = {a:g}" for a in alphas])
    ax1.set_ylabel("Mean EAS = Σ (score/100)²")
    ax1.set_ylim(0, max(steer + base) * 1.3)
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_title("Overall emotion strength", loc="left", fontsize=12)

    # right: per-emotion judge score at the strongest positive alpha
    a = max(alphas)
    d = runs[a]
    x2 = np.arange(len(judge_emotions))
    ax2.bar(x2 - w / 2, [d[f"baseline_{e}"].mean() for e in judge_emotions], w, color=BASE_C, label="Baseline")
    ax2.bar(x2 + w / 2, [d[f"enhanced_{e}"].mean() for e in judge_emotions], w,
            color=[COLORS.get(e, "#888") for e in judge_emotions])
    ax2.set_xticks(x2, [e.capitalize() for e in judge_emotions])
    ax2.set_ylabel("Mean judge score (0–100)")
    ax2.set_title(f"Which emotions the steering adds (α = {a:g})", loc="left", fontsize=12)
    ax2.text(0.01, 0.97, "grey = baseline · coloured = EV-steered", transform=ax2.transAxes,
             fontsize=10, color="#555", va="top")
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.1)

    titles(fig, "Additive emotion vectors — Emotion Absolute Strength (EAS)",
           "Qwen2.5-1.5B-Instruct · averaged 'base' emotion vector · LLM judge scores each emotion 0–100")
    fig.savefig(OUT / "ev_eas.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_conceptor()
    plot_ev_alpha()
    plot_eas()
    print(f"Wrote plots to {OUT}")
