#!/usr/bin/env python3
"""
Manual-review files from a scored sweep (shift_conceptor_sweep.py output).

    python make_review.py sweep_results_2emo            # -> review/2emo/
    python make_review.py sweep_results_5emo_gpt        # per-emotion subfolders are merged

Selected systems per emotion: unsteered baseline, project EVs (every alpha), and the
best config of each (method, data source) under the perplexity budget and the
cross-prompt overlap cap (excludes configs that answer every prompt with one template).

Writes:
    best_vs_additive_per_prompt.csv  one row per (emotion, prompt): baseline, best project EV,
                                     best additive mean shift and best shift+conceptor side by
                                     side, each with its output, confidence, perplexity, config
    side_by_side_<emotion>.md        every selected system's reply to each query — for reading
    rating_sheet.csv                 the same replies, shuffled and blinded, with empty rating
                                     columns (emotion / fluency / relevance, 1-5) — for scoring
    rating_key.csv                   id -> system, config and automatic scores (keep closed while rating)
"""

import random
import sys
from pathlib import Path

import pandas as pd

MAX_PPL = 50
MAX_OVERLAP = 0.10   # cross-prompt 3-gram overlap cap; see shift_conceptor_sweep.cross_prompt_overlap


def eligible(s):
    ok = s.ppl_median <= MAX_PPL
    if "overlap" in s:
        ok &= s.overlap <= MAX_OVERLAP
    return s[ok]


def load(res):
    """scored.csv + summary.csv from a results folder, or merged from its per-emotion subfolders."""
    dirs = [res] if (res / "scored.csv").exists() else sorted(p.parent for p in res.glob("*/scored.csv"))
    if not dirs:
        raise SystemExit(f"no scored.csv under {res}")
    return (pd.concat(pd.read_csv(d / "scored.csv") for d in dirs),
            pd.concat(pd.read_csv(d / "summary.csv") for d in dirs))


def best_within_budget(s, method, emotion):
    g = eligible(s[(s.method == method) & (s.emotion == emotion)])
    return None if g.empty else g.loc[g.conf.idxmax()]


def write_comparison(sc, s, out):
    """Wide table: per prompt, the best config of each method next to the baseline."""
    systems = {"baseline": None, "project_ev": "best project EV", "shift": "best additive shift",
               "shift_conc": "best shift+conceptor"}
    rows = []
    for e in s.emotion.unique():
        picks = {"baseline": "baseline"}
        for m in ["project_ev", "shift", "shift_conc"]:
            b = best_within_budget(s, m, e)
            if b is not None:
                picks[m] = b.config
        for q, qg in sc[sc.emotion == e].groupby("query", sort=False):
            row = {"emotion": e, "query": q}
            for m, cfg in picks.items():
                r = qg[(qg.method == m) & (qg.config == cfg)]
                if r.empty:
                    continue
                r = r.iloc[0]
                row[f"{m}_config"] = cfg
                row[f"{m}_output"] = str(r.response).replace("\n", " ")
                row[f"{m}_conf"] = round(r.target_conf, 3)
                row[f"{m}_ppl"] = round(r.ppl, 1)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "best_vs_additive_per_prompt.csv", index=False)
    means = {m: (df[f"{m}_conf"].mean(), df[f"{m}_ppl"].median())
             for m in systems if f"{m}_conf" in df}
    print("per-prompt comparison (mean conf, median ppl):",
          {m: (round(c, 3), round(p, 1)) for m, (c, p) in means.items()})


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "sweep_results_2emo")
    out = Path(__file__).resolve().parent / "review" / res.name.replace("sweep_results_", "")
    out.mkdir(parents=True, exist_ok=True)
    sc, s = load(res)
    write_comparison(sc, s, out)

    ok = eligible(s[s.method != "project_ev"])
    best = ok.loc[ok.groupby(["method", "source", "emotion"]).conf.idxmax()]
    picks = {e: [("baseline", "baseline")] +
                sorted({("project_ev", c) for c in s[(s.method == "project_ev") & (s.emotion == e)].config}) +
                [(r.method, r.config) for r in best[best.emotion == e].itertuples()]
             for e in s.emotion.unique()}

    rows = []
    for e, systems in picks.items():
        md = [f"# {e.capitalize()} — side-by-side outputs\n",
              f"Systems: baseline, project EVs, and the best config per method/source with "
              f"median perplexity ≤{MAX_PPL} and cross-prompt overlap ≤{MAX_OVERLAP}.\n"]
        for q, qg in sc[sc.emotion == e].groupby("query", sort=False):
            md.append(f"\n## {q}\n")
            for m, c in systems:
                r = qg[(qg.method == m) & (qg.config == c)]
                if not len(r):
                    continue
                r = r.iloc[0]
                text = str(r.response).replace("\n", " ")
                md.append(f"- **{m} · {c}** — conf {r.target_conf:.2f}, ppl {r.ppl:.0f}\n  > {text}\n")
                rows.append(dict(emotion=e, query=q, response=text, method=m, config=c,
                                 target_conf=round(r.target_conf, 3), ppl=round(r.ppl, 1)))
        (out / f"side_by_side_{e}.md").write_text("\n".join(md))

    df = pd.DataFrame(rows)
    order = list(range(len(df)))
    random.Random(0).shuffle(order)
    df = df.iloc[order].reset_index(drop=True)
    df.insert(0, "id", [f"R{i:04d}" for i in range(len(df))])
    sheet = df[["id", "emotion", "query", "response"]].copy()
    sheet["target_emotion_1to5"] = ""
    sheet["fluency_1to5"] = ""
    sheet["relevance_1to5"] = ""
    sheet["notes"] = ""
    sheet.to_csv(out / "rating_sheet.csv", index=False)
    df[["id", "emotion", "method", "config", "target_conf", "ppl"]].to_csv(out / "rating_key.csv", index=False)
    print(f"{len(df)} responses from {sum(map(len, picks.values()))} systems -> {out}")


if __name__ == "__main__":
    main()
