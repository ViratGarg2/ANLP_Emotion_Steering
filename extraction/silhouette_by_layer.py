#!/usr/bin/env python3
"""
Layerwise silhouette score of the emotion labels — one line, one point per layer.

Answers "at which depth does the model separate the emotions?" in a single
chart, instead of eyeballing 29 t-SNE panels.

Two possible input sources
--------------------------
embeddings (preferred)
    emotion_vectors/per_prompt/*.npz written by
    `qwen_emotion_extraction.py --mode per_prompt`. Silhouette is computed in
    the high-dimensional activation space (after the same L2 + PCA
    preprocessing tsne_emotion_vectors.py uses), so the numbers match the
    values printed in that script's panel subtitles.

tsne (fallback)
    tsne_plots/layerwise/layer_*.csv — the 2-D t-SNE coordinates.

    IMPORTANT: t-SNE does not preserve distances, so a silhouette computed on
    its output measures how separated the *picture* looks, not how separated
    the activations are. It is systematically optimistic and depends on the
    t-SNE seed. Useful for describing the figures; do not report it as a
    property of the model. The chart labels whichever space was used.

Usage
    python silhouette_by_layer.py                      # auto-detect source
    python silhouette_by_layer.py --source embeddings  # force high-dim
    python silhouette_by_layer.py --per_emotion        # add one line per emotion
    python silhouette_by_layer.py --space emotional
"""

import argparse
import csv
import glob
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

# sklearn's randomized SVD emits spurious divide/overflow RuntimeWarnings from
# the Accelerate BLAS on macOS + NumPy 2.x; results are unaffected.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="sklearn.utils.extmath")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# --------------------------------------------------------------------------
# Design tokens (light mode) — kept identical to tsne_emotion_vectors.py so the
# two figures read as one set.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e8e7e3"
ACCENT = "#2a78d6"

EMOTIONS = ["joy", "anger", "disgust", "fear", "sadness"]

EMOTION_COLOR = {
    "joy": "#eda100",      # yellow
    "anger": "#e34948",    # red
    "disgust": "#008300",  # green
    "fear": "#4a3aa7",     # violet
    "sadness": "#2a78d6",  # blue
}
EMOTION_MARKER = {
    "joy": "o", "anger": "s", "disgust": "^", "fear": "D", "sadness": "v",
}

HALO = [pe.withStroke(linewidth=3.0, foreground=SURFACE)]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_from_npz(per_prompt_dir, space, max_per_emotion, pca_dims, seed):
    """
    Yield (layer_index, X, labels) from the high-dimensional per-prompt set,
    applying the same preprocessing as tsne_emotion_vectors.py.
    """
    per_prompt_dir = Path(per_prompt_dir)
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
        sys.exit(f"error: no *_per_prompt.npz in {per_prompt_dir}")

    names = list(data)
    labels = np.concatenate([[n] * data[n].shape[0] for n in names])
    n_layers = min(v.shape[1] for v in data.values())

    for k in range(n_layers):
        X = np.concatenate([data[n][:, k, :] for n in names], axis=0).astype(np.float64)
        # L2-normalise, matching how the evaluator normalises at injection time
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        X = X / norms
        if pca_dims and X.shape[1] > pca_dims and X.shape[0] > 2:
            n = min(pca_dims, X.shape[0] - 1, X.shape[1])
            X = PCA(n_components=n, random_state=seed).fit_transform(X)
        yield k, X, labels


def load_from_csv(csv_dir):
    """Yield (layer_index, coords_2d, labels) from the t-SNE coordinate CSVs."""
    paths = sorted(glob.glob(str(Path(csv_dir) / "layer_*.csv")))
    if not paths:
        sys.exit(f"error: no layer_*.csv in {csv_dir}\n"
                 f"       run tsne_emotion_vectors.py first")
    print(f"  {len(paths)} layer CSVs in {csv_dir}")

    for path in paths:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"  warning: {Path(path).name} is empty, skipping")
            continue
        labels = np.array([r["emotion"] for r in rows])
        coords = np.array([[float(r["x"]), float(r["y"])] for r in rows])
        # prefer the recorded index; fall back to parsing the filename
        try:
            k = int(rows[0]["hidden_states_index"])
        except (KeyError, ValueError):
            m = re.search(r"layer_(\d+)", Path(path).stem)
            k = int(m.group(1)) if m else -1
        yield k, coords, labels


# --------------------------------------------------------------------------

def layer_label(key, n_blocks, mode):
    """Map a raw hidden_states index to a display label."""
    if mode == "raw":
        return f"layer_{key}"
    if key == 0:
        return "embed"
    if n_blocks is not None and key == n_blocks:
        return f"blk {key - 1}*"
    return f"blk {key - 1}"


def compute(source_iter, metric, per_emotion):
    """
    Returns
        layers:   list[int]
        overall:  list[float]  silhouette over all emotions
        by_emo:   dict[emotion -> list[float]]  mean silhouette of that
                  emotion's own samples (empty when per_emotion is False)
    """
    layers, overall, by_emo = [], [], {e: [] for e in EMOTIONS}

    for k, X, labels in source_iter:
        try:
            s = float(silhouette_score(X, labels, metric=metric))
        except ValueError:
            s = float("nan")
        layers.append(k)
        overall.append(s)

        if per_emotion:
            try:
                samples = silhouette_samples(X, labels, metric=metric)
            except ValueError:
                samples = np.full(len(labels), np.nan)
            for e in EMOTIONS:
                mask = labels == e
                by_emo[e].append(float(np.mean(samples[mask])) if mask.any()
                                 else float("nan"))

        print(f"  layer {k:2d}  silhouette {s:+.4f}")

    if per_emotion:
        by_emo = {e: v for e, v in by_emo.items() if v and not all(np.isnan(v))}
    else:
        by_emo = {}
    return layers, overall, by_emo


# --------------------------------------------------------------------------

def plot(layers, overall, by_emo, args, n_blocks, space_desc, out_png):
    fig, ax = plt.subplots(figsize=(10.0, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    # zero = no separation at all; the reference that makes the curve readable
    ax.axhline(0.0, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)

    if by_emo:
        for emotion, vals in by_emo.items():
            ax.plot(layers, vals, color=EMOTION_COLOR[emotion],
                    marker=EMOTION_MARKER[emotion], markersize=4.5,
                    linewidth=1.4, alpha=0.85, zorder=2,
                    markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=emotion.capitalize())
        # the all-emotion curve stays visually dominant
        ax.plot(layers, overall, color=INK, linewidth=2.6, zorder=4,
                label="All emotions")
    else:
        ax.plot(layers, overall, color=ACCENT, linewidth=2.4, zorder=3)
        ax.scatter(layers, overall, color=ACCENT, s=32, zorder=4,
                   edgecolors=SURFACE, linewidths=1.4)

    # annotate the peak
    finite = [v for v in overall if not np.isnan(v)]
    if finite:
        best = int(np.nanargmax(overall))
        lo, hi = min(finite), max(finite)
        ax.scatter([layers[best]], [overall[best]], s=150, facecolor="none",
                   edgecolors=INK if by_emo else ACCENT, linewidths=2.0, zorder=6)
        high = (overall[best] - lo) > 0.6 * ((hi - lo) or 1.0)
        # the peak usually sits at the right-hand end; point the callout inward
        # so it cannot run into the legend (which lives outside the right edge
        # in --per_emotion mode) or off the canvas
        rightish = layers[best] > layers[len(layers) // 2]
        ax.annotate(
            f"peak: {layer_label(layers[best], n_blocks, args.layer_labels)} "
            f"({overall[best]:+.3f})",
            (layers[best], overall[best]), textcoords="offset points",
            xytext=(-12 if rightish else 12, -22 if high else 12),
            ha="right" if rightish else "left",
            fontsize=10, fontweight="600",
            color=INK, zorder=7, path_effects=HALO)

    ax.set_xlabel("hidden_states index" if args.layer_labels == "raw"
                  else "layer", color=INK_2, fontsize=10)
    ax.set_ylabel("silhouette score", color=INK_2, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    step = 1 if len(layers) <= 16 else 2
    ax.set_xticks(layers[::step])

    ax.set_title("Which layers separate the emotions",
                 color=INK, fontsize=13, fontweight="600", loc="left", pad=30)
    ax.text(0.0, 1.012, space_desc, transform=ax.transAxes, color=INK_2,
            fontsize=9.5, va="bottom", ha="left")

    if by_emo:
        leg = ax.legend(frameon=False, fontsize=9, loc="upper left",
                        bbox_to_anchor=(1.01, 1.0), labelcolor=INK_2,
                        title="Silhouette of")
        leg.get_title().set_color(INK_2)
        leg.get_title().set_fontsize(9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # companion table view
    out_csv = out_png.with_suffix(".csv")
    header = ["layer_label", "hidden_states_index", "silhouette_all"] + \
             [f"silhouette_{e}" for e in by_emo]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, k in enumerate(layers):
            row = [layer_label(k, n_blocks, args.layer_labels), k,
                   "" if np.isnan(overall[i]) else f"{overall[i]:.6f}"]
            for e in by_emo:
                v = by_emo[e][i]
                row.append("" if np.isnan(v) else f"{v:.6f}")
            w.writerow(row)

    print(f"\nsaved {out_png}")
    print(f"saved {out_csv}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Layerwise silhouette score of the emotion labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = Path(__file__).resolve().parent
    p.add_argument("--source", choices=["auto", "embeddings", "tsne"], default="auto",
                   help="'embeddings' uses the per-prompt .npz (high-dim, the "
                        "meaningful number); 'tsne' uses the 2-D coordinate "
                        "CSVs; 'auto' prefers embeddings when present")
    p.add_argument("--per_prompt_dir",
                   default=str(here / "emotion_vectors" / "per_prompt"),
                   help="directory holding the *_per_prompt.npz set")
    p.add_argument("--csv_dir", default=str(here / "tsne_plots" / "layerwise"),
                   help="directory holding the per-layer t-SNE coordinate CSVs")
    p.add_argument("--out", default=None,
                   help="output PNG (default: <csv_dir>/../silhouette_by_layer.png)")
    p.add_argument("--space", choices=["diff", "emotional", "neutral"], default="diff",
                   help="which stored array to use (embeddings source only)")
    p.add_argument("--metric", choices=["auto", "cosine", "euclidean"], default="auto",
                   help="distance metric; 'auto' = cosine for embeddings, "
                        "euclidean for 2-D t-SNE coordinates")
    p.add_argument("--pca_dims", type=int, default=50,
                   help="PCA pre-reduction for the embeddings source; 0 disables")
    p.add_argument("--max_per_emotion", type=int, default=0,
                   help="subsample this many queries per emotion; 0 = all")
    p.add_argument("--per_emotion", action="store_true",
                   help="also draw one line per emotion (mean silhouette of "
                        "that emotion's own samples)")
    p.add_argument("--layer_labels", choices=["raw", "corrected"], default="raw",
                   help="'raw' prints the array index; 'corrected' maps index k "
                        "to decoder block k-1")
    p.add_argument("--seed", type=int, default=0, help="random_state for PCA")
    args = p.parse_args()

    have_npz = any(Path(args.per_prompt_dir).glob("*_per_prompt.npz"))
    source = args.source
    if source == "auto":
        source = "embeddings" if have_npz else "tsne"
    if source == "embeddings" and not have_npz:
        sys.exit(f"error: --source embeddings but no *_per_prompt.npz in "
                 f"{args.per_prompt_dir}\n"
                 f"       run: qwen_emotion_extraction.py --mode per_prompt")

    metric = args.metric
    if metric == "auto":
        metric = "cosine" if source == "embeddings" else "euclidean"

    n_blocks = None
    manifest = Path(args.per_prompt_dir) / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            n_layers = json.load(f).get("n_layers")
        if n_layers:
            n_blocks = n_layers - 1

    print(f"source: {source} ({metric})")
    if source == "embeddings":
        it = load_from_npz(args.per_prompt_dir, args.space,
                           args.max_per_emotion, args.pca_dims, args.seed)
        space_desc = (f"emotion labels in the {args.space} activation space "
                      f"({metric}, PCA {args.pca_dims}d)  ·  higher = better separated")
    else:
        print("  NOTE: measuring the 2-D t-SNE coordinates, not the activations.")
        print("        t-SNE distorts distances, so these values are optimistic")
        print("        and seed-dependent. Use --source embeddings to report a")
        print("        property of the model.")
        it = load_from_csv(args.csv_dir)
        space_desc = ("emotion labels in the 2-D t-SNE projection "
                      f"({metric})  ·  t-SNE distorts distances — optimistic, "
                      "seed-dependent")

    layers, overall, by_emo = compute(it, metric, args.per_emotion)
    if not layers:
        sys.exit("error: no layers produced a score")

    out_png = Path(args.out) if args.out else \
        Path(args.csv_dir).parent / "silhouette_by_layer.png"
    plot(layers, overall, by_emo, args, n_blocks, space_desc, out_png)

    finite = [v for v in overall if not np.isnan(v)]
    if finite:
        best = int(np.nanargmax(overall))
        worst = int(np.nanargmin(overall))
        print()
        print(f"  best  {layer_label(layers[best], n_blocks, args.layer_labels)}"
              f"  {overall[best]:+.4f}")
        print(f"  worst {layer_label(layers[worst], n_blocks, args.layer_labels)}"
              f"  {overall[worst]:+.4f}")


if __name__ == "__main__":
    main()
