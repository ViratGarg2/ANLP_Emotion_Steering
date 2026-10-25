#!/usr/bin/env python3
"""
LLM-judge evaluation of the held-out benchmark outputs (benchmark/test/scored.csv).

Each unique (prompt, reply) pair is scored once, blind to the steering method and to the
target emotion: coherence, relevance to the prompt, and the intensity of each of the five
emotions. The judge prompt is documented in EVALUATION_AND_METHODS.md.

    python llm_judge.py score --model gpt-5.5            # resumable; writes benchmark/judge/judgements.jsonl
    python llm_judge.py report                           # tables + paired bootstrap CIs

Requires OPENAI_API_KEY in qwen_testing/.env.
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from env_config import require_env  # noqa: E402

EMOTIONS = ["joy", "anger", "disgust", "fear", "sadness"]
JUDGE_DIR = HERE / "benchmark" / "judge"

SYSTEM_PROMPT = """You are a careful human annotator evaluating replies written by an AI assistant.
You will see the user's message and the assistant's reply. Rate the reply on the scales below,
the way an attentive native English reader would. Judge only what is on the page.

1. coherence (1-10): Is the reply fluent, grammatical, well-formed English that reads as a sensible
   response to the user's message? 10 = completely natural and clear. 5 = understandable but with
   clear errors, odd phrasing or partial nonsense. 1 = gibberish, word salad, broken tokens,
   non-English characters, or the same word/phrase repeated over and over.
2. relevance (1-10): Does the reply engage with what the user actually said or asked?
   10 = directly addresses it. 5 = loosely related. 1 = ignores the message entirely.
3. emotion intensity (0-10 for EACH of joy, anger, disgust, fear, sadness): How strongly does the
   reply itself express that emotion, through its tone, word choice and imagery?
   0 = not present at all. 3 = mild hint. 6 = clearly present. 10 = overwhelming, unmistakable.
   Rate each emotion independently; several can be present at once, and a neutral reply scores 0
   on all five. Do not reward a reply for merely naming an emotion word; judge the emotion
   actually conveyed. If the reply is unintelligible (coherence <= 2), rate every emotion 0.

Return ONLY a JSON object with exactly these keys:
{"coherence": <1-10>, "relevance": <1-10>, "joy": <0-10>, "anger": <0-10>, "disgust": <0-10>,
 "fear": <0-10>, "sadness": <0-10>}"""

USER_TEMPLATE = "User message:\n\"\"\"\n{query}\n\"\"\"\n\nAssistant reply:\n\"\"\"\n{response}\n\"\"\""

KEYS = ["coherence", "relevance"] + EMOTIONS


def judge_one(item, model, api_key):
    body = {"model": model, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": USER_TEMPLATE.format(**item)}]}
    if model.startswith(("gpt-5", "o")):
        body["reasoning_effort"] = "low"
    else:
        body["temperature"] = 0
    for attempt in range(5):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", timeout=120,
                              headers={"Authorization": f"Bearer {api_key}"}, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            out = json.loads(r.json()["choices"][0]["message"]["content"])
            if all(isinstance(out.get(k), (int, float)) for k in KEYS):
                return {"id": item["id"], "judge_model": model, **{k: float(out[k]) for k in KEYS}}
        except (requests.RequestException, ValueError, KeyError, IndexError):
            time.sleep(2)
    return None


def score(args):
    sc = pd.read_csv(HERE / "benchmark/test/scored.csv")
    uniq = sc.drop_duplicates(["query", "response"])[["query", "response"]].reset_index(drop=True)
    uniq["response"] = uniq.response.fillna("").astype(str)
    uniq["id"] = [f"J{i:05d}" for i in range(len(uniq))]
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    uniq.to_csv(JUDGE_DIR / "items.csv", index=False)
    out_path = JUDGE_DIR / "judgements.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [r for r in uniq.to_dict("records") if r["id"] not in done]
    random.Random(0).shuffle(todo)   # interleave methods/emotions so drift can't favour one system
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(uniq)} unique replies, {len(done)} already judged, judging {len(todo)} with {args.model}")
    api_key = require_env("OPENAI_API_KEY")
    t0, n_ok = time.time(), 0
    with open(out_path, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(lambda it: judge_one(it, args.model, api_key), todo)):
            if res:
                f.write(json.dumps(res) + "\n")
                f.flush()
                n_ok += 1
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"judged {n_ok}/{len(todo)} -> {out_path}")


def bootstrap_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    means = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(args):
    sc = pd.read_csv(HERE / "benchmark/test/scored.csv")
    sc["response"] = sc.response.fillna("").astype(str)
    items = pd.read_csv(JUDGE_DIR / "items.csv")
    items["response"] = items.response.fillna("").astype(str)
    j = pd.DataFrame([json.loads(l) for l in open(JUDGE_DIR / "judgements.jsonl")]).drop_duplicates("id")
    df = sc.merge(items, on=["query", "response"]).merge(j, on="id")
    df["target"] = [r[e] for r, e in zip(df.to_dict("records"), df.emotion)]
    others = lambda r: np.mean([r[e] for e in EMOTIONS if e != r["emotion"]])
    df["off_target"] = [others(r) for r in df.to_dict("records")]
    # emotion delivered in usable text: target intensity scaled by coherence (both 0-10 -> 0-10)
    df["target_x_coh"] = df.target * df.coherence / 10
    df.to_csv(JUDGE_DIR / "judged_benchmark.csv", index=False)
    print(f"merged {len(df)}/{len(sc)} benchmark rows with judgements ({df.judge_model.iloc[0]})")

    methods = ["baseline", "project_ev", "shift", "conceptor_paper", "shift_conc"]
    cols = ["target", "coherence", "relevance", "target_x_coh", "off_target"]
    tab = df.groupby(["emotion", "method"])[cols].mean().round(2).unstack("method")
    pd.set_option("display.width", 250)
    for c in cols:
        t = tab[c][[m for m in methods if m in tab[c]]]
        t.loc["MEAN"] = t.mean()
        print(f"\n== {c} (mean over 100 held-out prompts)")
        print(t.round(2).to_string())
    tab.to_csv(JUDGE_DIR / "judge_table.csv")

    rows = []
    for metric in ["target", "target_x_coh", "coherence"]:
        for rival in ["conceptor_paper", "shift", "project_ev", "baseline"]:
            pooled = []
            for e in EMOTIONS + ["ALL"]:
                es = EMOTIONS if e == "ALL" else [e]
                a = df[(df.method == "shift_conc") & df.emotion.isin(es)].set_index(["emotion", "query"])[metric]
                b = df[(df.method == rival) & df.emotion.isin(es)].set_index(["emotion", "query"])[metric]
                d = (a - b).dropna()
                if d.empty:
                    continue
                m, lo, hi = bootstrap_ci(d)
                rows.append({"metric": metric, "rival": rival, "emotion": e, "diff": m, "ci_lo": lo,
                             "ci_hi": hi, "win": float((d > 0).mean()), "tie": float((d == 0).mean()), "n": len(d)})
    p = pd.DataFrame(rows).round(3)
    print("\n== paired difference per prompt: shift+conceptor minus rival (mean, 95% bootstrap CI)")
    print(p[p.emotion == "ALL"].to_string(index=False))
    print("\n(per-emotion rows are in judge_paired.csv)")
    p.to_csv(JUDGE_DIR / "judge_paired.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", default="gpt-5.5")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int, default=0, help="judge only this many (for a trial run)")
    sub.add_parser("report")
    args = ap.parse_args()
    {"score": lambda: score(args), "report": lambda: report(args)}[args.cmd]()


if __name__ == "__main__":
    main()
