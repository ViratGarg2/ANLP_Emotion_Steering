"""
Plots for conceptor_emotion/shift_conceptor_sweep.py results.

    python plots/plot_sweep.py conceptor_emotion/sweep_results_2emo

Writes into that results directory:
    sweep_best_per_method.png   best config per method/data source, per emotion
    sweep_tradeoff.png          emotion confidence vs degenerate-output rate, every config
    sweep_heatmap_<source>.png  shift+conceptor: lambda x aperture per layer set

"Best" = highest valid confidence (degenerate outputs count as 0) among configs whose
degenerate rate is at most 10%.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("conceptor_emotion/sweep_results_2emo")
MAX_DEGEN = 0.10

plt.rcParams.update({
    "figure.facecolor": "#FCFCFA", "axes.facecolor": "#FCFCFA",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "axes.axisbelow": True, "font.size": 11,
})
SOURCE_LABEL = {"model_gen": "model's own gens", "synthetic": "gpt-5.5 pairs", "goemotions": "GoEmotions"}
COLORS = {"baseline": "#A0A4AB", "project_ev": "#2E86AB",
          "shift": "#E8A33D", "shift_conc": "#D1495B"}


def titles(fig, title, subtitle):
    fig.text(0.05, 0.965, title, fontsize=16, fontweight="bold", va="top")
    fig.text(0.05, 0.905, subtitle, fontsize=11, color="#555555", va="top")


def label(row):
    if row.method == "project_ev":
        return "project EV"
    name = "shift" if row.method == "shift" else "shift+conceptor"
    return f"{name}\n({SOURCE_LABEL.get(row.source, row.source)})"


def best_per_method(s):
    ok = s[s.degen_rate <= MAX_DEGEN]
    keys = ["method", "source", "emotion"]
    return ok.loc[ok.groupby(keys, dropna=False).valid_conf.idxmax()].reset_index(drop=True)


def plot_best(s):
    best = best_per_method(s)
    emotions = list(dict.fromkeys(s.emotion))
    fig, axes = plt.subplots(1, len(emotions), figsize=(7.5 * len(emotions), 7.2), squeeze=False)
    fig.subplots_adjust(top=0.80, bottom=0.25, left=0.05, right=0.98, wspace=0.15)
    for ax, e in zip(axes[0], emotions):
        b = best[best.emotion == e].copy()
        b["order"] = b.method.map({"project_ev": 0, "shift": 1, "shift_conc": 2})
        b = b.sort_values(["order", "source"])
        base = s[s.emotion == e].baseline_conf.iloc[0]
        x = np.arange(len(b) + 1)
        vals = [base] + b.valid_conf.tolist()
        cols = [COLORS["baseline"]] + [COLORS[m] for m in b.method]
        ax.bar(x, vals, color=cols, width=0.7)
        names = ["baseline"] + [label(r) for r in b.itertuples()]
        ax.set_xticks(x, names, rotation=35, ha="right", fontsize=9.5)
        for xi, r in zip(x[1:], b.itertuples()):
            cfg = r.config.replace(f"{r.source} ", "") if isinstance(r.source, str) else r.config
            ax.text(xi, r.valid_conf + 0.015, f"{cfg}\ndegen {r.degen_rate:.0%} · ppl {r.ppl_median:.0f}",
                    ha="center", fontsize=7.5, color="#333")
        ax.set_title(e.capitalize(), loc="left", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.3 + 0.05)
        ax.set_ylabel("Target-emotion confidence (degenerate → 0)")
    titles(fig, "Emotion steering — best configuration per method and data source",
           f"Qwen2.5-1.5B-Instruct · BART-MNLI confidence on held-out queries · configs with ≤{MAX_DEGEN:.0%} "
           "degenerate outputs · labels: config, degenerate rate, median perplexity")
    fig.savefig(RES / "sweep_best_per_method.png", dpi=180)
    plt.close(fig)


def plot_tradeoff(s):
    emotions = list(dict.fromkeys(s.emotion))
    fig, axes = plt.subplots(1, len(emotions), figsize=(7 * len(emotions), 6), squeeze=False)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.06, right=0.98, wspace=0.15)
    markers = {"model_gen": "o", "synthetic": "s", "goemotions": "^"}
    for ax, e in zip(axes[0], emotions):
        g = s[s.emotion == e]
        for (m, src), gg in g.groupby(["method", "source"], dropna=False):
            ax.scatter(gg.degen_rate, gg.conf, s=36, alpha=0.75, color=COLORS[m],
                       marker=markers.get(src, "D"),
                       label=("project EV" if m == "project_ev" else
                              f"{'shift' if m == 'shift' else 'shift+conc'} · {SOURCE_LABEL.get(src, src)}"))
        ax.axhline(g.baseline_conf.iloc[0], color="#888", ls=":", lw=1)
        ax.axvline(MAX_DEGEN, color="#B3261E", ls=":", lw=1)
        ax.set_xlabel("Degenerate-output rate")
        ax.set_ylabel("Mean target-emotion confidence")
        ax.set_title(e.capitalize(), loc="left", fontsize=13, fontweight="bold")
    axes[0][-1].legend(frameon=False, fontsize=8.5, loc="lower right")
    titles(fig, "Steering strength vs coherence — every configuration",
           "Top-left is best: high confidence with few degenerate outputs · dotted grey = unsteered baseline · "
           f"dotted red = {MAX_DEGEN:.0%} degeneracy cap")
    fig.savefig(RES / "sweep_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_heatmaps(s):
    sc = s[s.method == "shift_conc"]
    for src, g in sc.groupby("source"):
        layer_sets = sorted(g.layers.unique(), key=lambda t: int(t[1:].split("-")[0]))
        fig, axes = plt.subplots(1, len(layer_sets), figsize=(5.2 * len(layer_sets), 5), squeeze=False)
        fig.subplots_adjust(top=0.78, bottom=0.12, left=0.06, right=0.97, wspace=0.3)
        for ax, ls in zip(axes[0], layer_sets):
            piv = (g[g.layers == ls].groupby(["aperture", "lam"]).valid_conf.mean().unstack())
            im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=max(0.01, sc.valid_conf.max()), aspect="auto")
            ax.set_xticks(range(piv.shape[1]), [f"{v:g}" for v in piv.columns])
            ax.set_yticks(range(piv.shape[0]), [f"{v:g}" for v in piv.index])
            ax.set_xlabel("λ")
            ax.set_ylabel("aperture")
            ax.grid(False)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
            ax.set_title(ls, loc="left", fontsize=12)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="valid confidence")
        titles(fig, f"Shift+conceptor sweep — fitted on {SOURCE_LABEL.get(src, src)}",
               "Mean over emotions of target confidence with degenerate outputs scored 0")
        fig.savefig(RES / f"sweep_heatmap_{src}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    s = pd.read_csv(RES / "summary.csv")
    plot_best(s)
    plot_tradeoff(s)
    plot_heatmaps(s)
    print(f"wrote plots to {RES}")
