#!/usr/bin/env python3
"""
Per-layer t-SNE of Qwen2.5-1.5B emotion vectors.

One plot per layer. Every query is one point, coloured by its emotion.

Input is the per-prompt .npz set written by
    qwen_emotion_extraction.py --mode per_prompt   (or --mode both)
which stores one row per query instead of the query-averaged vector. The
averaged JSON files cannot be used here: they hold a single vector per
(emotion, layer), which is 5 points per layer, and t-SNE is meaningless there.

Output
    tsne_plots/layer_00.png ... layer_28.png     one panel per layer
    tsne_plots/layer_00.csv ...                  the same points as a table

Spaces (--space)
    diff       (default) emotional - neutral, per query
    emotional  raw emotion-conditioned states
    neutral    raw neutral states
  Each emotion is run on a DIFFERENT query set, so the raw spaces separate by
  query topic as much as by emotion. `diff` cancels most of that. Raw-space
  plots are marked topic-confounded in the subtitle.

Layer labels
    hidden_states has N+1 entries: index 0 is the embedding output and index i
    is the output of decoder block i-1. Labels print the raw array index by
    default; --layer_labels corrected shows the decoder block mapping.

Usage
    python tsne_emotion_vectors.py
    python tsne_emotion_vectors.py --space emotional
    python tsne_emotion_vectors.py --layers 10,14,20 --max_per_emotion 50
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

# scikit-learn's randomized SVD triggers spurious divide/overflow RuntimeWarnings
# from the Accelerate BLAS on macOS + NumPy 2.x. Results are unaffected
# (verified finite), but they bury the real progress output.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="sklearn.utils.extmath")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------------------
# Design tokens (light mode)
#
# The 5 hues are a validated all-pairs (scatter) palette on surface #fcfcfb:
# worst all-pairs CVD dE 7.2, normal-vision dE 16.3. dE 7.2 sits in the 6-8
# band, which is legal ONLY alongside secondary encoding -- hence the
# per-emotion marker shapes and the direct centroid labels. Dark mode is
# deliberately absent: no 5-hue assignment clears the all-pairs floors on a
# dark surface.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e8e7e3"

EMOTIONS = ["joy", "anger", "disgust", "fear", "sadness"]

EMOTION_COLOR = {
    "joy": "#eda100",      # yellow
    "anger": "#e34948",    # red
    "disgust": "#008300",  # green
    "fear": "#4a3aa7",     # violet
    "sadness": "#2a78d6",  # blue
}
EMOTION_MARKER = {
    "joy": "o",
    "anger": "s",
    "disgust": "^",
    "fear": "D",
    "sadness": "v",
}

HALO = [pe.withStroke(linewidth=3.0, foreground=SURFACE)]


# --------------------------------------------------------------------------

def load_per_prompt(per_prompt_dir, space, max_per_emotion=0):
    """
    Load per-query hidden states from the .npz set.

    Only the requested `space` array is read out of each archive, so memory
    stays at a third of the file set.

    Returns
        data:     dict[emotion -> np.ndarray(n_queries, n_layers, hidden_dim)]
        n_blocks: decoder block count implied by the manifest, or None
    """
    per_prompt_dir = Path(per_prompt_dir)
    if not per_prompt_dir.is_dir():
        sys.exit(f"error: no per-prompt directory at {per_prompt_dir}\n"
                 f"       run: qwen_emotion_extraction.py --mode per_prompt")

    data = {}
    for emotion in EMOTIONS:
        path = per_prompt_dir / f"Qwen25_1B5_{emotion}_per_prompt.npz"
        if not path.exists():
            print(f"  warning: missing {path.name}, skipping {emotion}")
            continue
        with np.load(path, allow_pickle=True) as archive:
            if space not in archive:
                sys.exit(f"error: '{space}' not in {path.name} "
                         f"(has: {', '.join(archive.files)})")
            arr = np.asarray(archive[space], dtype=np.float32)
        if max_per_emotion and arr.shape[0] > max_per_emotion:
            arr = arr[:max_per_emotion]
        data[emotion] = arr
        print(f"  {emotion:8s} {arr.shape[0]:4d} queries x {arr.shape[1]} layers "
              f"x {arr.shape[2]} dims")

    if not data:
        sys.exit(f"error: no *_per_prompt.npz files in {per_prompt_dir}\n"
                 f"       run: qwen_emotion_extraction.py --mode per_prompt")

    n_blocks = None
    manifest = per_prompt_dir / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            n_layers = json.load(f).get("n_layers")
        if n_layers:
            n_blocks = n_layers - 1

    return data, n_blocks


def layer_label(key, n_blocks, mode):
    """Map a raw hidden_states index to a display label."""
    if mode == "raw":
        return f"layer_{key}"
    if key == 0:
        return "embed"
    if n_blocks is not None and key == n_blocks:
        return f"blk {key - 1}*"   # * = post final-norm
    return f"blk {key - 1}"


def prepare(X, normalize):
    X = np.asarray(X, dtype=np.float64)
    if normalize == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
    return X


def project(X, metric, seed, pca_dims):
    """PCA pre-reduction (standard practice, faster and less noisy) then t-SNE."""
    if pca_dims and X.shape[1] > pca_dims and X.shape[0] > 2:
        n = min(pca_dims, X.shape[0] - 1, X.shape[1])
        Xr = PCA(n_components=n, random_state=seed).fit_transform(X)
    else:
        Xr = X

    perplexity = max(2.0, min(30.0, (Xr.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        init="pca" if metric == "euclidean" else "random",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    )
    return tsne.fit_transform(Xr), Xr, f"t-SNE (perplexity {perplexity:.0f}, {metric})"


def declutter_labels(centroids, extent, min_frac=0.16):
    """
    Nudge overlapping centroid labels apart.

    On a layer with no separation every centroid lands in the middle and the
    labels stack into an unreadable pile. Labels sit at their centroid where
    that is legible, and are pushed radially outward when they would collide
    with one already placed.
    """
    diag = float(np.hypot(*extent)) or 1.0
    min_dist = min_frac * diag
    cx = float(np.mean([c[1] for c in centroids]))
    cy = float(np.mean([c[2] for c in centroids]))

    placed = []
    for i, (name, x, y) in enumerate(centroids):
        dx, dy = x - cx, y - cy
        if np.hypot(dx, dy) < 1e-9:
            angle = 2.0 * np.pi * i / max(len(centroids), 1)
            dx, dy = np.cos(angle), np.sin(angle)
        norm = np.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm

        px, py = x, y
        for _ in range(12):
            if all(np.hypot(px - qx, py - qy) >= min_dist for _, qx, qy in placed):
                break
            px += ux * min_dist * 0.5
            py += uy * min_dist * 0.5
        placed.append((name, px, py))

    return placed


def plot_layer(coords, labels, counts, k, sil, method, n_blocks, args, out_dir):
    """One panel: every query a point, coloured and shaped by emotion."""
    names = list(counts)
    total = sum(counts.values())

    fig, ax = plt.subplots(figsize=(8.4, 6.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    # t-SNE axis units are arbitrary, so the tick values carry no meaning
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel("t-SNE 1", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("t-SNE 2", color=INK_MUTED, fontsize=9)

    for name in names:
        mask = labels == name
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=EMOTION_COLOR[name], marker=EMOTION_MARKER[name],
                   s=34, alpha=0.75, zorder=3,
                   edgecolors=SURFACE, linewidths=0.6)

    # Direct labels -- identity never rests on colour alone, as the CVD warn
    # band requires.
    centroids = [(n, coords[labels == n, 0].mean(), coords[labels == n, 1].mean())
                 for n in names]
    extent = (np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
    for (name, lx, ly), (_, cx, cy) in zip(declutter_labels(centroids, extent),
                                           centroids):
        if np.hypot(lx - cx, ly - cy) > 1e-9:
            ax.plot([cx, lx], [cy, ly], color=INK_MUTED, linewidth=0.9,
                    alpha=0.6, zorder=5)
        ax.annotate(name.capitalize(), (lx, ly),
                    fontsize=11, fontweight="700", color=INK,
                    ha="center", va="center", zorder=6, path_effects=HALO)

    handles = [
        Line2D([0], [0], marker=EMOTION_MARKER[n], linestyle="none",
               markerfacecolor=EMOTION_COLOR[n], markeredgecolor=SURFACE,
               markeredgewidth=1.0, markersize=9,
               label=f"{n.capitalize()}  (n={counts[n]})")
        for n in names
    ]
    leg = ax.legend(handles=handles, frameon=False, fontsize=9.5,
                    loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    labelcolor=INK_2, title="Emotion")
    leg.get_title().set_color(INK_2)
    leg.get_title().set_fontsize(9.5)

    label = layer_label(k, n_blocks, args.layer_labels)
    sil_txt = "n/a" if np.isnan(sil) else f"{sil:+.3f}"
    confound = "" if args.space == "diff" else \
        "  ·  topic-confounded: each emotion uses a different query set"

    # pad leaves room for the subtitle between title and plot area
    ax.set_title(f"Emotion separation at {label}", color=INK, fontsize=13,
                 fontweight="600", loc="left", pad=30)
    ax.text(0.0, 1.012,
            f"{total} query vectors ({args.space}), {method}  ·  "
            f"silhouette {sil_txt}{confound}",
            transform=ax.transAxes, color=INK_2, fontsize=9.5,
            va="bottom", ha="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"layer_{k:02d}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # companion table view
    with open(png.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["emotion", "layer_label", "hidden_states_index", "x", "y"])
        w.writerows([[labels[i], label, k, f"{coords[i,0]:.4f}", f"{coords[i,1]:.4f}"]
                     for i in range(len(labels))])

    print(f"  layer {k:2d}  silhouette {sil_txt}  ->  {png.name}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Per-layer t-SNE of per-query emotion vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = Path(__file__).resolve().parent
    p.add_argument("--per_prompt_dir", default=str(here / "emotion2" / "per_prompt"),
                   help="directory holding the *_per_prompt.npz set")
    p.add_argument("--output_dir", default=str(here / "tsne_plots"),
                   help="directory for the PNG + CSV output")
    p.add_argument("--space", choices=["diff", "emotional", "neutral"], default="diff",
                   help="which stored array to project; raw spaces are "
                        "confounded by query topic")
    p.add_argument("--layers", default="",
                   help="comma-separated layer indices to plot; empty = all")
    p.add_argument("--normalize", choices=["l2", "none"], default="l2",
                   help="L2-normalise each vector first (matches how the "
                        "evaluator normalises at injection time)")
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine",
                   help="distance metric for t-SNE and the silhouette score")
    p.add_argument("--pca_dims", type=int, default=50,
                   help="PCA pre-reduction before t-SNE; 0 disables")
    p.add_argument("--max_per_emotion", type=int, default=0,
                   help="subsample this many queries per emotion; 0 = all")
    p.add_argument("--layer_labels", choices=["raw", "corrected"], default="raw",
                   help="'raw' prints the array index; 'corrected' maps index k "
                        "to decoder block k-1")
    p.add_argument("--seed", type=int, default=0, help="random_state for t-SNE/PCA")
    args = p.parse_args()

    out_dir = Path(args.output_dir)

    print("loading per-prompt vectors")
    data, n_blocks = load_per_prompt(args.per_prompt_dir, args.space,
                                     args.max_per_emotion)

    names = list(data)
    counts = {n: data[n].shape[0] for n in names}
    n_layers = min(v.shape[1] for v in data.values())
    labels = np.concatenate([[n] * counts[n] for n in names])

    if args.layers:
        wanted = sorted({int(x) for x in args.layers.split(",") if x.strip() != ""})
        bad = [k for k in wanted if not 0 <= k < n_layers]
        if bad:
            sys.exit(f"error: layer(s) {bad} out of range 0..{n_layers - 1}")
    else:
        wanted = list(range(n_layers))

    print(f"\n{sum(counts.values())} points per layer, {len(wanted)} layers, "
          f"{args.space} space")

    for k in wanted:
        X = prepare(np.concatenate([data[n][:, k, :] for n in names], axis=0),
                    args.normalize)
        coords, Xr, method = project(X, args.metric, args.seed, args.pca_dims)

        # silhouette on the reduced space, not the 2-D embedding: it measures
        # real separation rather than a t-SNE artefact
        try:
            sil = float(silhouette_score(Xr, labels, metric=args.metric))
        except ValueError:
            sil = float("nan")

        plot_layer(coords, labels, counts, k, sil, method, n_blocks, args, out_dir)

    print(f"\ndone — {len(wanted)} panels in {out_dir}")


if __name__ == "__main__":
    main()
