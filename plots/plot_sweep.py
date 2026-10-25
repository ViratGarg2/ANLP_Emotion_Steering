"""
Plots for conceptor_emotion/shift_conceptor_sweep.py results.

    python plots/plot_sweep.py conceptor_emotion/sweep_results_2emo

Writes into that results directory:
    sweep_best_per_method.png   best config per method/data source, per emotion
    sweep_frontier.png          best confidence reachable within a perplexity budget
    sweep_tradeoff.png          emotion confidence vs median perplexity, every config
    sweep_heatmap_<source>.png  shift+conceptor: lambda x aperture per layer set

"Best" = highest mean target-emotion confidence among configs whose median perplexity is
at most MAX_PPL and whose cross-prompt overlap is at most MAX_OVERLAP. The budget stands in for coherence: the classifier also rewards
incoherent emotion-keyword soup, which shows up as high perplexity.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("conceptor_emotion/sweep_results_2emo")
MAX_PPL = 50      # median perplexity budget; above it, outputs drift into emotion-keyword soup
MAX_OVERLAP = 0.10   # cross-prompt 3-gram overlap cap (unsteered ~0.02); above it the same
                     # emotional template comes back for every prompt


def eligible(s):
    """Configs that are both fluent enough and still prompt-specific."""
    ok = s.ppl_median <= MAX_PPL
    if "overlap" in s:
        ok &= s.overlap <= MAX_OVERLAP
    return s[ok]

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
    ok = eligible(s)
    keys = ["method", "source", "emotion"]
    return ok.loc[ok.groupby(keys, dropna=False).conf.idxmax()].reset_index(drop=True)


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
        vals = [base] + b.conf.tolist()
        cols = [COLORS["baseline"]] + [COLORS[m] for m in b.method]
        ax.bar(x, vals, color=cols, width=0.7)
        names = ["baseline"] + [label(r) for r in b.itertuples()]
        ax.set_xticks(x, names, rotation=35, ha="right", fontsize=9.5)
        for xi, r in zip(x[1:], b.itertuples()):
            cfg = r.config.replace(f"{r.source} ", "") if isinstance(r.source, str) else r.config
            ax.text(xi, r.conf + 0.015, f"{cfg}\nppl {r.ppl_median:.0f}",
                    ha="center", fontsize=7.5, color="#333")
        ax.set_title(e.capitalize(), loc="left", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.3 + 0.05)
        ax.set_ylabel("Target-emotion confidence")
    titles(fig, "Emotion steering — best configuration per method and data source",
           f"Qwen2.5-1.5B-Instruct · BART-MNLI confidence on held-out queries · median perplexity ≤{MAX_PPL}, "
           f"cross-prompt overlap ≤{MAX_OVERLAP} · labels: config, median perplexity")
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
            ax.scatter(gg.ppl_median, gg.conf, s=30, alpha=0.7, color=COLORS[m],
                       marker=markers.get(src, "D"),
                       label=("project EV" if m == "project_ev" else
                              f"{'shift' if m == 'shift' else 'shift+conc'} · {SOURCE_LABEL.get(src, src)}"))
        ax.axhline(g.baseline_conf.iloc[0], color="#888", ls=":", lw=1)
        ax.axvline(MAX_PPL, color="#B3261E", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Median perplexity (log)")
        ax.set_ylabel("Mean target-emotion confidence")
        ax.set_title(e.capitalize(), loc="left", fontsize=13, fontweight="bold")
    axes[0][-1].legend(frameon=False, fontsize=8.5, loc="lower right")
    titles(fig, "Emotion strength vs fluency — every configuration",
           "Top-left is best: high confidence at low perplexity · dotted grey = unsteered baseline · "
           f"dotted red = perplexity budget {MAX_PPL}")
    fig.savefig(RES / "sweep_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_frontier(s):
    """Best confidence reachable within a median-perplexity budget, per method/source."""
    emotions = list(dict.fromkeys(s.emotion))
    caps = np.geomspace(4, 400, 40)
    ok = s[s.overlap <= MAX_OVERLAP] if "overlap" in s else s
    fig, axes = plt.subplots(1, len(emotions), figsize=(7 * len(emotions), 6), squeeze=False)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.06, right=0.98, wspace=0.15)
    styles = {"model_gen": ":", "synthetic": "-", "goemotions": "--"}
    for ax, e in zip(axes[0], emotions):
        g = ok[ok.emotion == e]
        for (m, src), gg in g.groupby(["method", "source"], dropna=False):
            y = [gg[gg.ppl_median <= c].conf.max() if (gg.ppl_median <= c).any() else np.nan for c in caps]
            lab = ("project EV" if m == "project_ev" else
                   f"{'shift' if m == 'shift' else 'shift+conc'} · {SOURCE_LABEL.get(src, src)}")
            ax.step(caps, y, where="post", color=COLORS[m], ls=styles.get(src, "-"),
                    lw=2.6 if src == "synthetic" or m == "project_ev" else 1.6, label=lab)
        ax.axhline(s[s.emotion == e].baseline_conf.iloc[0], color="#888", ls=":", lw=1)
        ax.axvline(MAX_PPL, color="#B3261E", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Median perplexity budget (log)")
        ax.set_ylabel("Best confidence within budget")
        ax.set_ylim(0, 1.02)
        ax.set_title(e.capitalize(), loc="left", fontsize=13, fontweight="bold")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower right")
    titles(fig, "Emotion strength vs fluency budget",
           "Higher-left is better · dotted grey = unsteered · dotted red = budget used for the bar chart · "
           f"templated configs (overlap > {MAX_OVERLAP}) excluded")
    fig.savefig(RES / "sweep_frontier.png", dpi=180)
    plt.close(fig)


def plot_heatmaps(s):
    """Mean confidence over emotions per (aperture, lambda); cells whose mean median perplexity
    exceeds the budget are greyed out and show their perplexity instead."""
    sc = s[s.method == "shift_conc"]
    for src, g in sc.groupby("source"):
        layer_sets = sorted(g.layers.unique(), key=lambda t: (int(t[1:].split("-")[0]), t))
        ncol = min(5, len(layer_sets))
        nrow = -(-len(layer_sets) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow + 1.2), squeeze=False)
        fig.subplots_adjust(top=1 - 1.25 / (4.2 * nrow + 1.2), bottom=0.07, left=0.05, right=0.92,
                            wspace=0.35, hspace=0.45)
        vmax = max(0.01, g[g.ppl_median <= MAX_PPL].conf.max() if (g.ppl_median <= MAX_PPL).any() else 0.01)
        for ax, ls in zip(axes.flat, layer_sets):
            gl = g[g.layers == ls]
            conf = gl.groupby(["aperture", "lam"]).conf.mean().unstack()
            ppl = gl.groupby(["aperture", "lam"]).ppl_median.mean().unstack()
            ov = (gl.groupby(["aperture", "lam"]).overlap.mean().unstack() if "overlap" in gl
                  else ppl * 0)
            templ = ov.values > MAX_OVERLAP
            im = ax.imshow(np.where((ppl.values <= MAX_PPL) & ~templ, conf.values, np.nan),
                           cmap="RdYlGn", vmin=0, vmax=vmax, aspect="auto")
            ax.set_facecolor("#D9D9D9")
            ax.set_xticks(range(conf.shape[1]), [f"{v:g}" for v in conf.columns], fontsize=8)
            ax.set_yticks(range(conf.shape[0]), [f"{v:g}" for v in conf.index], fontsize=8)
            ax.set_xlabel("λ")
            ax.set_ylabel("aperture")
            ax.grid(False)
            for i in range(conf.shape[0]):
                for j in range(conf.shape[1]):
                    over = ppl.values[i, j] > MAX_PPL
                    label = (f"p{ppl.values[i, j]:.0f}" if over else
                             f"t{ov.values[i, j]:.2f}" if templ[i, j] else f"{conf.values[i, j]:.2f}")
                    ax.text(j, i, label, ha="center", va="center", fontsize=7,
                            color="#666" if over or templ[i, j] else "black")
            ax.set_title(ls, loc="left", fontsize=11)
        for ax in list(axes.flat)[len(layer_sets):]:
            ax.axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="mean target confidence")
        titles(fig, f"Shift+conceptor sweep — fitted on {SOURCE_LABEL.get(src, src)}",
               f"Mean over emotions · grey cells exceed the perplexity budget ({MAX_PPL}, shown as p…) or "
               f"are templated across prompts (overlap > {MAX_OVERLAP}, shown as t…)")
        fig.savefig(RES / f"sweep_heatmap_{src}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    s = pd.read_csv(RES / "summary.csv")
    plot_best(s)
    plot_frontier(s)
    plot_tradeoff(s)
    plot_heatmaps(s)
    print(f"wrote plots to {RES}")
