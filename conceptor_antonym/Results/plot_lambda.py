#!/usr/bin/env python3
"""
Plot layer vs. accuracy for the conceptor-lambda sweep on Qwen2.5-1.5B-Instruct.

For each layer, picks the BEST (beta, aperture, lambda_) configuration based on
the "Average" row's final_accuracy across experiment repeats. Annotates each
point with the chosen hyperparameters.

Steering formula:  h' = h + lambda * (beta * C - I) h
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CSV_PATH = HERE / "Qwen2.5-1.5B-Instruct_capitalize_conceptor_lambda.csv"
OUT_PNG  = HERE / "plot_capitalize_conceptor_lambda_layers.png"


def best_per_layer(path: Path) -> pd.DataFrame:
    """For each layer, return the (beta, aperture, lambda_) row with max Average accuracy."""
    df = pd.read_csv(path)
    avg = df[df["experiment"].astype(str) == "Average"].copy()
    avg["final_accuracy"] = pd.to_numeric(avg["final_accuracy"], errors="coerce")
    avg = avg.dropna(subset=["final_accuracy"])
    idx = avg.groupby("layer")["final_accuracy"].idxmax()
    best = avg.loc[idx, ["layer", "beta", "aperture", "lambda_", "final_accuracy"]]
    return best.sort_values("layer").reset_index(drop=True)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    best = best_per_layer(CSV_PATH)
    print("Best conceptor (β, α, λ) per layer:")
    print(best.to_string(index=False))

    _, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        best["layer"], best["final_accuracy"],
        marker="s", linestyle="-", linewidth=1.8, color="tab:blue",
        label="Conceptor (best β, α, λ per layer)",
    )

    for _, r in best.iterrows():
        ax.annotate(
            f"β={r['beta']:g}\nα={r['aperture']:g}\nλ={r['lambda_']:g}",
            xy=(r["layer"], r["final_accuracy"]),
            xytext=(0, 9), textcoords="offset points",
            fontsize=7, ha="center", color="tab:blue",
        )

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Capitalize — Qwen2.5-1.5B-Instruct\n"
        r"Layer vs. accuracy  ($h'=h+\lambda(\beta C - I)h$, best config per layer)"
    )
    ax.set_xticks(sorted(best["layer"].unique()))
    ax.set_ylim(bottom=0.0, top=max(0.05, best["final_accuracy"].max() * 1.25))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved plot to {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
