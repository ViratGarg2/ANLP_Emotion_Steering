#!/usr/bin/env python3
"""
Plot layer vs. accuracy for capitalize task on Qwen2.5-1.5B-Instruct.

For each layer, picks the BEST hyperparameter config:
  - baseline:                no layer dimension -> drawn as a horizontal line
  - addition:                best beta per layer
  - conceptor:               best (beta, aperture) per layer
  - controllable conceptor:  best (beta, aperture, lambda) per layer
                             (steering formula h' = h + lambda*(beta*C - I)h)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BASELINE_CSV   = HERE / "Qwen2.5-1.5B-Instruct_capitalize_baseline.csv"
ADDITION_CSV   = HERE / "Qwen2.5-1.5B-Instruct_capitalize_addition_1.csv"
CONCEPTOR_CSV  = HERE / "Qwen2.5-1.5B-Instruct_capitalize_conceptor_3.csv"
CONTROLLABLE_CSV = HERE / "Qwen2.5-1.5B-Instruct_capitalize_conceptor_6.csv"
OUT_PNG        = HERE / "plot_capitalize_layers.png"


def load_baseline_overall(path: Path) -> float:
    """Return overall baseline accuracy as a fraction in [0, 1]."""
    df = pd.read_csv(path)
    overall = df[df["experiment"].astype(str) == "Overall"]
    if not overall.empty:
        acc = float(overall.iloc[0]["baseline_accuracy"])
    else:
        # Fallback: average the per-experiment rows
        per_exp = df[pd.to_numeric(df["experiment"], errors="coerce").notna()]
        acc = float(per_exp["baseline_accuracy"].mean())
    # Stored as a percentage in baseline CSV; normalise to fraction if so
    return acc / 100.0 if acc > 1.0 else acc


def best_per_layer_addition(path: Path) -> pd.DataFrame:
    """For addition: pick the (beta) with highest Average accuracy per layer."""
    df = pd.read_csv(path)
    avg = df[df["experiment"].astype(str) == "Average"].copy()
    avg["final_accuracy"] = pd.to_numeric(avg["final_accuracy"], errors="coerce")
    idx = avg.groupby("layer")["final_accuracy"].idxmax()
    best = avg.loc[idx, ["layer", "beta", "final_accuracy"]].sort_values("layer")
    return best.reset_index(drop=True)


def best_per_layer_conceptor(path: Path) -> pd.DataFrame:
    """For conceptor: pick the (beta, aperture) with highest Average accuracy per layer."""
    df = pd.read_csv(path)
    avg = df[df["experiment"].astype(str) == "Average"].copy()
    avg["final_accuracy"] = pd.to_numeric(avg["final_accuracy"], errors="coerce")
    idx = avg.groupby("layer")["final_accuracy"].idxmax()
    best = avg.loc[idx, ["layer", "beta", "aperture", "final_accuracy"]].sort_values("layer")
    return best.reset_index(drop=True)


def best_per_layer_controllable(path: Path) -> pd.DataFrame:
    """For controllable conceptor: pick the (beta, aperture, lambda_) with highest Average per layer."""
    df = pd.read_csv(path)
    avg = df[df["experiment"].astype(str) == "Average"].copy()
    avg["final_accuracy"] = pd.to_numeric(avg["final_accuracy"], errors="coerce")
    avg = avg.dropna(subset=["final_accuracy"])
    idx = avg.groupby("layer")["final_accuracy"].idxmax()
    best = avg.loc[idx, ["layer", "beta", "aperture", "lambda_", "final_accuracy"]].sort_values("layer")
    return best.reset_index(drop=True)


def main():
    baseline_acc = load_baseline_overall(BASELINE_CSV)
    best_add     = best_per_layer_addition(ADDITION_CSV)
    best_conc    = best_per_layer_conceptor(CONCEPTOR_CSV)
    best_ctrl    = best_per_layer_controllable(CONTROLLABLE_CSV)

    print(f"Baseline overall accuracy: {baseline_acc:.4f}")
    print("\nBest addition per layer:")
    print(best_add.to_string(index=False))
    print("\nBest conceptor per layer:")
    print(best_conc.to_string(index=False))
    print("\nBest controllable conceptor per layer:")
    print(best_ctrl.to_string(index=False))

    # ---------------- Plot ----------------
    _, ax = plt.subplots(figsize=(11, 6))

    # Baseline: horizontal line spanning the layer range we have data for
    layers_seen = sorted(
        set(best_add["layer"])
        .union(set(best_conc["layer"]))
        .union(set(best_ctrl["layer"]))
    )
    if layers_seen:
        ax.axhline(
            baseline_acc, color="grey", linestyle="--", linewidth=1.5,
            label=f"Baseline ({baseline_acc * 100:.2f}%)",
        )

    # Addition: best beta per layer
    ax.plot(
        best_add["layer"], best_add["final_accuracy"],
        marker="o", linestyle="-", linewidth=1.8,
        color="tab:orange", label="Addition (best β)",
    )
    for _, r in best_add.iterrows():
        ax.annotate(
            f"β={r['beta']:g}",
            xy=(r["layer"], r["final_accuracy"]),
            xytext=(0, -14), textcoords="offset points",
            fontsize=7, ha="center", color="tab:orange",
        )

    # Conceptor: best (beta, aperture) per layer
    ax.plot(
        best_conc["layer"], best_conc["final_accuracy"],
        marker="s", linestyle="-", linewidth=1.8,
        color="tab:blue", label="Conceptor (best β, α)",
    )
    for _, r in best_conc.iterrows():
        ax.annotate(
            f"β={r['beta']:g}\nα={r['aperture']:g}",
            xy=(r["layer"], r["final_accuracy"]),
            xytext=(0, 8), textcoords="offset points",
            fontsize=7, ha="center", color="tab:blue",
        )

    # Controllable conceptor: best (beta, aperture, lambda) per layer
    ax.plot(
        best_ctrl["layer"], best_ctrl["final_accuracy"],
        marker="D", linestyle="-", linewidth=1.8,
        color="tab:green", label="Controllable Conceptor (best β, α, λ)",
    )
    for _, r in best_ctrl.iterrows():
        ax.annotate(
            f"β={r['beta']:g}\nα={r['aperture']:g}\nλ={r['lambda_']:g}",
            xy=(r["layer"], r["final_accuracy"]),
            xytext=(0, 8), textcoords="offset points",
            fontsize=7, ha="center", color="tab:green",
        )

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Capitalize task — Qwen2.5-1.5B-Instruct\nLayer vs. accuracy (best config per layer)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)

    # X ticks at every layer we plotted
    if layers_seen:
        ax.set_xticks(layers_seen)
    ax.set_ylim(bottom=0.0)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved plot to {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
