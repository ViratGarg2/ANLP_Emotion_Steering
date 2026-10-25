#!/usr/bin/env python3
"""
Held-out benchmark: stacked shift+conceptor vs the best single-layer published conceptor (beta*C*h).

Selection happens on the sweep prompts (EmotionQuery+ [300, 350)); testing on prompts that were
used for neither fitting nor selection (EmotionQuery+ [250, 300) and [350, 400), 100 prompts).

    # 1. pick one config per (emotion, method) from the sweep summaries, by a fixed rule
    python benchmark_test.py select --out benchmark/selection.json
    # 2. regenerate only those configs on the held-out prompts and score them (GPU)
    python benchmark_test.py run --selection benchmark/selection.json --out_dir benchmark/test
    # 3. paired comparison with bootstrap confidence intervals
    python benchmark_test.py report --out_dir benchmark/test

Selection rule (same for every method): highest mean target-emotion confidence among configs
with median perplexity <= 50 and cross-prompt overlap <= 0.10. The published conceptor is
restricted to single layers ("one layer at a time", as in Postmus & Abreu / Triantafyllopoulos
et al.); shift+conceptor uses its layer blocks.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import shift_conceptor_sweep as S

HERE = Path(__file__).resolve().parent
EMOTIONS = ["joy", "anger", "disgust", "fear", "sadness"]
MAX_PPL, MAX_OVERLAP = 50, 0.10
TEST_SLICES = [(250, 300), (350, 400)]
METHODS = {   # method -> sweep result folders its candidates come from
    "project_ev": ["sweep_results_5emo_gpt"],
    "shift": ["sweep_results_5emo_gpt"],
    "shift_conc": ["sweep_results_5emo_gpt"],
    "conceptor_paper": ["sweep_results_5emo_paper", "sweep_results_5emo_bch_layers"],
}


def layer_range(name):
    a, b = name[1:].split("-")
    return list(range(int(a), int(b) + 1))


# ------------------------------------------------------------------------------ select

def select(out):
    picks = []
    for method, dirs in METHODS.items():
        frames = [pd.read_csv(p) for d in dirs for p in sorted((HERE / d).glob("*/summary.csv"))]
        if not frames:
            print(f"warning: no summaries for {method} in {dirs}")
            continue
        s = pd.concat(frames, ignore_index=True)   # several folders share row labels
        s = s[s.method == method]
        if method == "conceptor_paper":
            s = s[s.layers.map(lambda l: len(layer_range(l)) == 1)]
        ok = s[(s.ppl_median <= MAX_PPL) & (s.overlap <= MAX_OVERLAP)]
        for e in EMOTIONS:
            g = ok[ok.emotion == e]
            if g.empty:
                print(f"warning: no eligible {method} config for {e}")
                continue
            r = g.loc[g.conf.idxmax()]
            picks.append({k: (None if pd.isna(r[k]) else r[k]) for k in
                          ["method", "emotion", "config", "layers", "aperture", "lam", "alpha"]} |
                         {"sel_conf": round(float(r.conf), 4), "sel_ppl": round(float(r.ppl_median), 2),
                          "sel_overlap": round(float(r.overlap), 4),
                          "n_candidates": int((s.emotion == e).sum())})
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(picks, open(out, "w"), indent=1, default=float)
    print(pd.DataFrame(picks)[["emotion", "method", "config", "sel_conf", "sel_ppl", "n_candidates"]]
          .sort_values(["emotion", "method"]).to_string(index=False))
    print(f"wrote {out}")


# --------------------------------------------------------------------------------- run

def run(args):
    picks = json.load(open(args.selection))
    queries = json.load(open(S.ROOT / "conceptor_emotion/EmotionQuery+.json"))
    test_q = [q for a, b in TEST_SLICES for q in queries[a:b]][:args.n_test]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok, model = S.load_model(args.model_name, args.device, getattr(torch, args.dtype))
    prompts = [S.chat(tok, S.NEUTRAL, q) for q in test_q]
    steer, dev, rows = S.Steerer(model), args.device, []
    gen = lambda: S.generate(tok, model, prompts, args.gen_tokens, args.batch_size)

    def record(p, responses):
        rows.extend(dict(method=p["method"], source="synthetic", config=p["config"], emotion=p["emotion"],
                         query=q, response=r, layers=p["layers"] or "", lam=p["lam"],
                         aperture=p["aperture"], alpha=p["alpha"]) for q, r in zip(test_q, responses))

    t0 = time.time()
    base = gen()
    for e in EMOTIONS:
        rows.extend(dict(method="baseline", source="", config="baseline", emotion=e, query=q, response=r,
                         layers="", lam=None, aperture=None, alpha=None) for q, r in zip(test_q, base))
    items = lambda tone: S.dataset_items("synthetic", tone, args.n_fit_data)
    layers_all = sorted({l for p in picks if p["layers"] for l in layer_range(p["layers"])})
    Xn = S.collect_teacher_forced(tok, model, items("neutral"), layers_all, args.batch_size)
    mu_n = {l: Xn[l].mean(0) for l in layers_all}
    del Xn

    for e in EMOTIONS:
        mine = [p for p in picks if p["emotion"] == e]
        need = sorted({l for p in mine if p["layers"] for l in layer_range(p["layers"])})
        Xe = S.collect_teacher_forced(tok, model, items(e), need, args.batch_size) if need else {}
        mu_e = {l: Xe[l].mean(0) for l in need}
        for p in mine:
            m = p["method"]
            ls = layer_range(p["layers"]) if p["layers"] else []
            if m == "project_ev":
                steer.project_ev(S.load_project_ev(e, dev), p["alpha"])
            elif m == "shift":
                steer.shift_conceptor({l: (mu_e[l].to(dev), mu_n[l].to(dev), None) for l in ls}, p["lam"])
            elif m == "shift_conc":
                C = {l: S.conceptor(*S.eig_cov(Xe[l]), p["aperture"]).to(dev) for l in ls}
                steer.shift_conceptor({l: (mu_e[l].to(dev), mu_n[l].to(dev), C[l]) for l in ls}, p["lam"])
            elif m == "conceptor_paper":
                C = {l: S.conceptor(*S.eig_corr(Xe[l]), p["aperture"]).to(dev) for l in ls}
                steer.conceptor_paper(C, p["alpha"])
            record(p, gen())
            steer.clear()
        del Xe
        print(f"[{e}] done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "generations.csv", index=False)
    df = S.score(df, tok, model, args.device, args.batch_size)
    df.to_csv(out / "scored.csv", index=False)
    S.summarise(df).to_csv(out / "summary.csv", index=False)
    print(f"wrote {out}")


# ------------------------------------------------------------------------------ report

def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(args):
    out = Path(args.out_dir)
    sc = pd.read_csv(out / "scored.csv")
    s = pd.read_csv(out / "summary.csv")
    rows = []
    for e in EMOTIONS:
        g = s[s.emotion == e].set_index("method")
        r = {"emotion": e, "baseline": round(g.baseline_conf.iloc[0], 3)}
        for m in ["project_ev", "shift", "conceptor_paper", "shift_conc"]:
            if m in g.index:
                r[m] = round(g.loc[m, "conf"], 3)
                r[m + "_ppl"] = round(g.loc[m, "ppl_median"], 1)
                r[m + "_ov"] = round(g.loc[m, "overlap"], 3)
        rows.append(r)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("== held-out test set: mean target confidence (median perplexity, cross-prompt overlap)")
    print(t.to_string(index=False))

    print("\n== paired difference per prompt, shift+conceptor minus each rival (mean, 95% bootstrap CI)")
    diffs = []
    for rival in ["conceptor_paper", "shift", "project_ev"]:
        allp = []
        for e in EMOTIONS:
            a = sc[(sc.emotion == e) & (sc.method == "shift_conc")].set_index("query").target_conf
            b = sc[(sc.emotion == e) & (sc.method == rival)].set_index("query").target_conf
            d = (a - b).dropna()
            if d.empty:
                continue
            allp.append(d.values)
            m, lo, hi = bootstrap_ci(d)
            diffs.append({"rival": rival, "emotion": e, "diff": m, "ci_lo": lo, "ci_hi": hi,
                          "win_rate": float((d > 0).mean()), "n": len(d)})
        if allp:
            m, lo, hi = bootstrap_ci(np.concatenate(allp))
            diffs.append({"rival": rival, "emotion": "ALL", "diff": m, "ci_lo": lo, "ci_hi": hi,
                          "win_rate": float((np.concatenate(allp) > 0).mean()),
                          "n": int(sum(map(len, allp)))})
    d = pd.DataFrame(diffs).round(3)
    print(d.to_string(index=False))
    t.to_csv(out / "report_table.csv", index=False)
    d.to_csv(out / "report_paired.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--out", type=Path, default=HERE / "benchmark/selection.json")
    p = sub.add_parser("run")
    p.add_argument("--selection", default=str(HERE / "benchmark/selection.json"))
    p.add_argument("--out_dir", default=str(HERE / "benchmark/test"))
    p.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--n_fit_data", type=int, default=250)
    p.add_argument("--gen_tokens", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=50)
    p = sub.add_parser("report")
    p.add_argument("--out_dir", default=str(HERE / "benchmark/test"))
    args = ap.parse_args()
    {"select": lambda: select(args.out), "run": lambda: run(args), "report": lambda: report(args)}[args.cmd]()


if __name__ == "__main__":
    main()
