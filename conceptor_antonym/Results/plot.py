#!/usr/bin/env python3
"""
Layerwise best-accuracy comparison of the three steering methods.

For a given task, plots one line per method. Each point is that method's BEST
hyperparameter configuration at that layer, so the curve is an upper envelope
over the sweep rather than any single config.

  addition         h' = h + beta*v                      best over beta
  conceptor-paper  h' = beta_c * C * h                   best over (beta, aperture)
                   the published form, arXiv:2410.16314 eq. 4
  conceptor        h' = h + lambda*(beta*C - I)h         best over (beta, aperture, lambda)
                   residual generalisation; equals the paper form at lambda=1
  conceptor-mean   h' = h + lambda*((beta*C - I)h + b)   best over (beta, aperture, lambda)
                   with b = mu - beta*C*mu

The unsteered baseline, if present, is drawn as a horizontal reference.

Usage
    python plot.py --task capitalize
    python plot.py --task antonyms --results_dir ../Results_v2
    python plot.py --task capitalize --out my_plot.png
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# --------------------------------------------------------------------------
# Design tokens (light mode)
#
# The three series hues are categorical slots 1-3, the subset that clears the
# colour-vision-deficiency and normal-vision separation gates as a group. Aqua
# sits below 3:1 contrast on this surface, so the relief rule applies: every
# series carries a direct end-of-line label and a companion CSV, never colour
# alone.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e8e7e3"

METHODS = {
    "addition": {
        "suffix": "addition",
        "label": "Addition",
        "color": "#2a78d6",
        "marker": "o",
        "params": ["beta"],
    },
    "conceptor_paper": {
        "suffix": "conceptor_paper",
        "label": "Conceptor (paper, βCh)",
        "color": "#4a3aa7",
        "marker": "^",
        "params": ["beta", "aperture"],
    },
    "conceptor": {
        "suffix": "conceptor",
        "label": "Conceptor (λ-residual)",
        "color": "#eb6834",
        "marker": "s",
        "params": ["beta", "aperture", "lambda_"],
    },
    "conceptor_mc": {
        "suffix": "conceptor_mc",
        "label": "Conceptor + bias recovery",
        "color": "#1baf7a",
        "marker": "D",
        "params": ["beta", "aperture", "lambda_"],
    },
}

MODEL_PREFIX = "Qwen2.5-1.5B-Instruct"
HALO = [pe.withStroke(linewidth=3.0, foreground=SURFACE)]


# --------------------------------------------------------------------------

def find_csv(results_dir, task, suffix):
    """
    Locate a method's CSV, tolerating the _1/_2 collision suffixes that
    _get_filename adds when a base name is already taken. The most recent
    match by modification time wins.
    """
    base = f"{MODEL_PREFIX}_{task}_{suffix}"
    exact = results_dir / f"{base}.csv"
    if exact.exists():
        return exact
    matches = sorted(results_dir.glob(f"{base}_[0-9]*.csv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def best_per_layer(path, params):
    """
    Per layer, the config with the highest Average accuracy.

    Only `experiment == "Average"` rows are used — those are the mean across
    repeats that experiments.py appends after each config.
    """
    df = pd.read_csv(path)
    avg = df[df["experiment"].astype(str) == "Average"].copy()
    avg["final_accuracy"] = pd.to_numeric(avg["final_accuracy"], errors="coerce")
    avg = avg.dropna(subset=["final_accuracy"])
    if avg.empty:
        return None

    idx = avg.groupby("layer")["final_accuracy"].idxmax()
    cols = ["layer"] + [p for p in params if p in avg.columns] + ["final_accuracy"]
    return avg.loc[idx, cols].sort_values("layer").reset_index(drop=True)


def load_baseline(results_dir, task):
    """Unsteered accuracy as a fraction, or None if there is no baseline CSV."""
    path = find_csv(results_dir, task, "baseline")
    if path is None:
        return None
    df = pd.read_csv(path)
    overall = df[df["experiment"].astype(str) == "Overall"]
    if not overall.empty:
        acc = float(overall.iloc[0]["baseline_accuracy"])
    else:
        per_exp = df[pd.to_numeric(df["experiment"], errors="coerce").notna()]
        acc = float(per_exp["baseline_accuracy"].mean())
    # the baseline CSV stores a percentage; the others store fractions
    return acc / 100.0 if acc > 1.0 else acc


def config_string(row, params):
    """Compact 'β=2 α=0.05 λ=1' for the peak annotation."""
    glyph = {"beta": "β", "aperture": "α", "lambda_": "λ"}
    return " ".join(f"{glyph.get(p, p)}={row[p]:g}"
                    for p in params if p in row and pd.notna(row[p]))


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Layerwise best-accuracy plot for the three steering methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = Path(__file__).resolve().parent
    # New sweeps write to Results_v2; fall back to this script's own directory
    # for the older runs that live alongside it.
    default_results = here.parent / "Results_v2"
    if not default_results.is_dir():
        default_results = here

    p.add_argument("--task", required=True,
                   help="task name, e.g. capitalize or antonyms")
    p.add_argument("--results_dir", default=str(default_results),
                   help="directory holding the method CSVs")
    p.add_argument("--out", default=None,
                   help="output PNG (default: plot_<task>_layers.png in --results_dir)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_png = Path(args.out) if args.out else results_dir / f"plot_{args.task}_layers.png"

    # ---- load
    series = {}
    for key, spec in METHODS.items():
        path = find_csv(results_dir, args.task, spec["suffix"])
        if path is None:
            print(f"  missing: {MODEL_PREFIX}_{args.task}_{spec['suffix']}.csv — skipping")
            continue
        best = best_per_layer(path, spec["params"])
        if best is None or best.empty:
            print(f"  no Average rows in {path.name} — skipping "
                  f"(is the sweep still running?)")
            continue
        series[key] = best
        print(f"  loaded {path.name}: {len(best)} layers, "
              f"peak {best['final_accuracy'].max():.1%}")

    if not series:
        sys.exit(f"error: no usable CSVs for task '{args.task}' in {results_dir}")

    baseline_acc = load_baseline(results_dir, args.task)

    # ---- plot
    fig, ax = plt.subplots(figsize=(10.5, 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    if baseline_acc is not None:
        ax.axhline(baseline_acc, color=INK_MUTED, linestyle="--", linewidth=1.4, zorder=1)
        ax.text(0.004, baseline_acc, f" unsteered baseline {baseline_acc:.1%}",
                transform=ax.get_yaxis_transform(), va="bottom", ha="left",
                fontsize=9, color=INK_2, path_effects=HALO, zorder=4)

    for key, best in series.items():
        spec = METHODS[key]
        ax.plot(best["layer"], best["final_accuracy"],
                color=spec["color"], marker=spec["marker"], markersize=6,
                linewidth=2.0, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=3, label=spec["label"])

        # Direct label at the right end of each line — identity never rests on
        # colour alone.
        last = best.iloc[-1]
        ax.annotate(f" {spec['label']}",
                    (last["layer"], last["final_accuracy"]),
                    fontsize=9.5, fontweight="600", color=spec["color"],
                    va="center", ha="left", zorder=5, path_effects=HALO)

    # Selective annotation: each method's winning config at its peak layer
    # only, never a label on every point. Methods whose peaks land on the same
    # layer are stacked above the highest of them and named inline, so the
    # labels neither overlap nor rely on colour for attribution.
    peaks = []
    for key, best in series.items():
        row = best.loc[best["final_accuracy"].idxmax()]
        peaks.append({
            "spec": METHODS[key],
            "layer": row["layer"],
            "acc": float(row["final_accuracy"]),
            "cfg": config_string(row, METHODS[key]["params"]),
        })

    by_layer = {}
    for pk in peaks:
        by_layer.setdefault(pk["layer"], []).append(pk)

    max_stack = max(len(g) for g in by_layer.values())
    for layer, group in by_layer.items():
        group.sort(key=lambda d: -d["acc"])
        anchor = group[0]["acc"]
        for i, pk in enumerate(group):
            ax.scatter([layer], [pk["acc"]], s=130, facecolor="none",
                       edgecolors=pk["spec"]["color"], linewidths=1.8, zorder=5)
            ax.annotate(f"{pk['spec']['label']}  {pk['acc']:.1%}  {pk['cfg']}",
                        (layer, anchor), textcoords="offset points",
                        xytext=(0, 14 + i * 15), ha="center",
                        fontsize=8.5, color=INK, zorder=6, path_effects=HALO)

    all_layers = sorted({int(l) for b in series.values() for l in b["layer"]})
    ax.set_xticks(all_layers)
    ax.set_xlabel("Transformer layer (extraction and injection site)",
                  color=INK_2, fontsize=10)
    ax.set_ylabel("Accuracy (best config at that layer)", color=INK_2, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    # headroom scaled to the tallest stack of peak callouts, so they sit
    # inside the axes instead of running into the subtitle
    y_max = max(b["final_accuracy"].max() for b in series.values())
    ax.set_ylim(0.0, y_max * (1.14 + 0.07 * max_stack))
    ax.margins(x=0.10)

    ax.set_title(f"{args.task.capitalize()} — steering method comparison",
                 color=INK, fontsize=14, fontweight="600", loc="left", pad=30)
    ax.text(0.0, 1.012,
            f"{MODEL_PREFIX}  ·  best hyperparameter configuration per layer  ·  "
            f"{len(series)} methods",
            transform=ax.transAxes, color=INK_2, fontsize=9.5,
            va="bottom", ha="left")

    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left",
                    labelcolor=INK_2)
    leg.set_zorder(7)

    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # ---- companion table view
    rows = []
    for key, best in series.items():
        for _, r in best.iterrows():
            rows.append({
                "method": METHODS[key]["label"],
                "layer": int(r["layer"]),
                "accuracy": round(float(r["final_accuracy"]), 6),
                "config": config_string(r, METHODS[key]["params"]),
            })
    table = pd.DataFrame(rows).sort_values(["method", "layer"])
    out_csv = out_png.with_suffix(".csv")
    table.to_csv(out_csv, index=False)

    print(f"\nsaved {out_png}")
    print(f"saved {out_csv}")
    print()
    for key, best in series.items():
        peak = best.loc[best["final_accuracy"].idxmax()]
        print(f"  {METHODS[key]['label']:28s} peak {peak['final_accuracy']:6.1%} "
              f"at layer {int(peak['layer']):2d}  ({config_string(peak, METHODS[key]['params'])})")


if __name__ == "__main__":
    main()
