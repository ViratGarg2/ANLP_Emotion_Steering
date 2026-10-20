#!/usr/bin/env python3
"""
Sweep for the shift+conceptor emotion steering fix, compared against additive EVs.

Why the original conceptor_emotion_evaluation.py produced word salad:
  1. h += beta*C(h - mu) at all 28 layers is expansive (eigenvalues up to 1+beta),
     so the residual norm explodes (173 -> ~1e7 at layer 27).
  2. mu and C were fitted on the prompt's last token (the "start of reply" state),
     then applied to every generated token, which keeps predicting reply openers.
  3. C fitted from ~100 samples in 1536 dims is low-rank, so (C - I) ~ -I and any
     residual form erases the token's content.

Fix tested here — statistics from *generated* tokens, a few mid layers, and

    h' = h + lambda * [ (mu_e - mu_n) + (C_e - I)(h - mu_n) ]

i.e. the additive mean shift plus a conceptor reshaping of the deviation from the
neutral mean. With C = I (aperture -> inf) it reduces to the plain mean shift, which
is run as the ablation "shift".

Methods:
    baseline     no steering
    project_ev   repo EVs (extraction/emotion_vectors), unit-norm, all layers, injected
                 exactly like confidence_score_evaluator.py (layer + both layernorms)
    shift        h + lambda*(mu_e - mu_n)                    on a layer set
    shift_conc   the fix above                               on a layer set

Scoring: BART-large-MNLI zero-shot confidence for the target emotion (as in the repo),
degenerate-output rate, and base-model perplexity of the response.

Usage:
    python shift_conceptor_sweep.py --quick                      # end-to-end smoke test
    python shift_conceptor_sweep.py --out_dir sweep_results      # full sweep
    python shift_conceptor_sweep.py --out_dir sweep_results --score_only
"""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parent.parent
EMOTIONS = ["joy", "anger", "disgust", "fear", "sadness"]
LABELS = EMOTIONS + ["neutral"]
NEUTRAL = ("You are an AI assistant. Respond neutrally and objectively to the user's query "
           "without showing any particular emotion. Maintain a balanced and factual tone.")
EMOTION_PROMPT = ("Respond with strong {e} emotion. Express clear {e} in your tone, "
                  "word choice, and perspective.")

LAYER_SETS = {"L6-12": list(range(6, 13)), "L10-16": list(range(10, 17)), "L14-20": list(range(14, 21))}
CONC_LAMBDAS = [0.15, 0.3, 0.45, 0.6]
APERTURES = [0.05, 0.2, 1.0]
SHIFT_LAMBDAS = [0.3, 0.6, 1.0, 1.5]
EV_ALPHAS = [2.0, 4.0]


# --------------------------------------------------------------------------- model

def load_model(name, device, dtype):
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    # SDPA on MPS returns NaN logits for left-padded rows (any dtype); eager is correct there
    attn = "eager" if device == "mps" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn).to(device).eval()
    return tok, model


def chat(tok, system, query):
    return tok.apply_chat_template([{"role": "system", "content": system},
                                    {"role": "user", "content": query}],
                                   tokenize=False, add_generation_prompt=True)


def stop_ids(tok, model):
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids])
    ids.add(tok.eos_token_id)
    return ids


@torch.no_grad()
def generate(tok, model, prompts, max_new_tokens, batch_size):
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        outs += [tok.decode(s[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for s in g]
    return outs


# --------------------------------------------------------------- generated-token stats

@torch.no_grad()
def collect_generated_states(tok, model, system, queries, layers, n_tokens, batch_size):
    """Hidden states (layer outputs) of every generated token, up to the first EOS."""
    store = {l: [] for l in layers}
    step = {}

    def hook(l):
        def fn(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if h.shape[1] == 1:                       # decode step, not prefill
                step.setdefault(l, []).append(h[:, 0].float().cpu())
        return fn

    handles = [model.model.layers[l].register_forward_hook(hook(l)) for l in layers]
    stops = stop_ids(tok, model)
    try:
        for i in range(0, len(queries), batch_size):
            step.clear()
            prompts = [chat(tok, system, q) for q in queries[i:i + batch_size]]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            g = model.generate(**enc, max_new_tokens=n_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)[:, enc.input_ids.shape[1]:].cpu()
            # decode step k feeds generated token k-1; keep it while no stop token came before
            stopped = torch.zeros_like(g, dtype=torch.bool)
            for s in stops:
                stopped |= g == s
            alive = (stopped.cumsum(1) == 0)          # token k is a real token
            for l in layers:
                states = torch.stack(step[l], 1)      # (B, T-1, d)
                mask = alive[:, :states.shape[1]]
                store[l].append(states[mask])
    finally:
        for h in handles:
            h.remove()
    return {l: torch.cat(v) for l, v in store.items()}


@torch.no_grad()
def collect_teacher_forced(tok, model, items, layers, batch_size):
    """Hidden states at every response token when (query, response) is fed as a chat turn
    under the neutral system prompt — the same position and context steering acts on."""
    store = {l: [] for l in layers}
    cur = {}
    handles = [model.model.layers[l].register_forward_hook(
        lambda m, i, o, l=l: cur.__setitem__(l, (o[0] if isinstance(o, tuple) else o).float().cpu()))
        for l in layers]
    try:
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            prompts = [chat(tok, NEUTRAL, q) for q, _ in batch]
            fulls = [p + r for p, (_, r) in zip(prompts, batch)]
            n_resp = [len(tok(f).input_ids) - len(tok(p).input_ids) for f, p in zip(fulls, prompts)]
            model(**tok(fulls, return_tensors="pt", padding=True).to(model.device))
            for l in layers:
                h = cur[l]                                # (B, T, d), left-padded
                store[l] += [h[b, h.shape[1] - n:] for b, n in enumerate(n_resp) if n > 0]
    finally:
        for h in handles:
            h.remove()
    return {l: torch.cat(v) for l, v in store.items()}


GENERIC_QUERY = "Tell me what's on your mind."


def dataset_items(source, tone, n_fit):
    """(query, response) pairs for one tone from a built dataset (see build_emotion_datasets.py)."""
    data_dir = Path(__file__).resolve().parent / "data"
    if source == "synthetic":
        recs = [json.loads(l) for l in open(data_dir / "emotion_pairs_synthetic.jsonl")]
        return [(r["query"], r[tone]) for r in recs[:n_fit]]
    if source == "goemotions":
        texts = json.load(open(data_dir / "goemotions_emotion_texts.json"))[tone]
        return [(GENERIC_QUERY, t) for t in texts[:n_fit]]
    raise ValueError(source)


def eig_cov(X):
    mu = X.mean(0)
    R = (X - mu).T @ (X - mu) / X.shape[0]
    s, U = torch.linalg.eigh(R.double())
    return s.clamp(min=0).float(), U.float()


def conceptor(s, U, aperture):
    return (U * (s / (s + aperture ** -2))) @ U.T


# ----------------------------------------------------------------------- steering

class Steerer:
    def __init__(self, model):
        self.model, self.handles = model, []

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _last_token(self, module, fn):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            h = h.clone()
            h[:, -1] = fn(h[:, -1].float()).to(h.dtype)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        self.handles.append(module.register_forward_hook(hook))

    def shift_conceptor(self, layer_params, lam):
        """layer_params[l] = (mu_e, mu_n, C or None). C=None means C=I (plain mean shift)."""
        for l, (mu_e, mu_n, C) in layer_params.items():
            shift = mu_e - mu_n
            if C is None:
                fn = lambda h, shift=shift: h + lam * shift
            else:
                fn = lambda h, shift=shift, mu_n=mu_n, C=C: (
                    h + lam * (shift + (h - mu_n) @ C - (h - mu_n)))
            self._last_token(self.model.model.layers[l], fn)

    def project_ev(self, vectors, alpha):
        """Replicates confidence_score_evaluator.py: hooks every module whose name contains
        'layers', not 'mlp'/'attn', with <=3 dots (the layer and both layernorms), adding
        alpha * unit(vector[layer_index]) to the last token."""
        for name, module in self.model.named_modules():
            if "layers." in name and "mlp" not in name and "attn" not in name and name.count(".") <= 3:
                l = int(name.split("layers.")[1].split(".")[0])
                if l in vectors:
                    v = torch.nn.functional.normalize(vectors[l], dim=0)
                    self._last_token(module, lambda h, v=v: h + alpha * v)


def load_project_ev(emotion, device):
    d = json.load(open(ROOT / f"extraction/emotion_vectors/Qwen25_1B5_{emotion}_vectors.json"))
    return {int(k.split("_")[1]): torch.tensor(v, dtype=torch.float32, device=device)
            for k, v in d.items() if k.startswith("layer_")}


# ------------------------------------------------------------------------ scoring

def is_degenerate(text):
    if not isinstance(text, str) or len(text.strip()) < 5:
        return True
    if text.count(" ") / len(text) < 0.05:           # glued word salad: "OhWowAhOh..."
        return True
    words = text.lower().split()
    return len(words) >= 8 and len(set(words)) / len(words) < 0.4   # "the the the ..."


@torch.no_grad()
def perplexity(tok, model, prompts, responses, batch_size):
    """Base-model perplexity of each response given its (neutral) prompt."""
    out = []
    for i in range(0, len(prompts), batch_size):
        P, R = prompts[i:i + batch_size], responses[i:i + batch_size]
        full = tok([p + r for p, r in zip(P, R)], return_tensors="pt", padding=True).to(model.device)
        plen = [len(tok(p).input_ids) for p in P]
        logits = model(**full).logits.float()
        nll = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), full.input_ids[:, 1:],
                                                reduction="none")
        T = full.input_ids.shape[1]
        for b in range(len(P)):
            pad = int((full.attention_mask[b] == 0).sum())
            start = pad + plen[b] - 1                 # first response token is predicted here
            n = T - 1 - start
            out.append(math.exp(nll[b, start:].mean().item()) if n > 0 else float("nan"))
    return out


def score(df, tok, model, device, batch_size):
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
                   device=device if device != "cpu" else -1)
    texts = df["response"].fillna("").tolist()
    conf = [dict.fromkeys(LABELS, 0.0) for _ in texts]
    valid = [i for i, t in enumerate(texts) if t.strip()]
    for i, r in zip(valid, tqdm(clf([texts[i] for i in valid], LABELS, batch_size=32),
                                total=len(valid), desc="BART-MNLI")):
        conf[i] = dict(zip(r["labels"], r["scores"]))
    df["target_conf"] = [c[e] for c, e in zip(conf, df["emotion"])]
    for lab in LABELS:
        df[f"conf_{lab}"] = [c[lab] for c in conf]
    df["degenerate"] = df["response"].map(is_degenerate)
    prompts = [chat(tok, NEUTRAL, q) for q in df["query"]]
    df["ppl"] = perplexity(tok, model, prompts, texts, batch_size)
    return df


def summarise(df):
    base = df[df.method == "baseline"]
    rows = []
    for key, g in df[df.method != "baseline"].groupby(["method", "config", "emotion"], sort=False):
        b = base[base["query"].isin(g["query"])]
        rows.append({
            "method": key[0], "source": g.source.iloc[0], "config": key[1], "emotion": key[2],
            "layers": g.layers.iloc[0], "lam": g.lam.iloc[0], "aperture": g.aperture.iloc[0],
            "alpha": g.alpha.iloc[0], "n": len(g),
            "conf": g.target_conf.mean(),
            "baseline_conf": b[f"conf_{key[2]}"].mean(),
            "valid_conf": (g.target_conf * ~g.degenerate).mean(),   # degenerate outputs score 0
            "degen_rate": g.degenerate.mean(),
            "ppl_median": g.ppl.median(),
            "baseline_ppl_median": b.ppl.median(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--emotions", default=",".join(EMOTIONS))
    ap.add_argument("--queries_file", default=str(ROOT / "conceptor_emotion/EmotionQuery+.json"))
    ap.add_argument("--n_fit", type=int, default=30, help="queries used to fit mu/C (from the start)")
    ap.add_argument("--n_test", type=int, default=20, help="held-out queries (from index 300)")
    ap.add_argument("--fit_tokens", type=int, default=24)
    ap.add_argument("--gen_tokens", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "sweep_results"))
    ap.add_argument("--stats_sources", default="model_gen,synthetic,goemotions",
                    help="where mu/C are fitted from: model_gen, synthetic, goemotions")
    ap.add_argument("--n_fit_data", type=int, default=200, help="texts per tone for dataset sources")
    ap.add_argument("--quick", action="store_true", help="tiny end-to-end smoke test")
    ap.add_argument("--score_only", action="store_true", help="rescore an existing generations.csv")
    args = ap.parse_args()

    global LAYER_SETS, CONC_LAMBDAS, APERTURES, SHIFT_LAMBDAS, EV_ALPHAS
    emotions = args.emotions.split(",")
    sources = args.stats_sources.split(",")
    if args.quick:
        args.n_fit, args.n_test, args.fit_tokens, args.gen_tokens, args.n_fit_data = 3, 2, 8, 12, 4
        LAYER_SETS = {"L10-16": LAYER_SETS["L10-16"]}
        CONC_LAMBDAS, APERTURES, SHIFT_LAMBDAS, EV_ALPHAS = [0.3], [0.2], [1.0], [4.0]
        emotions = ["joy", "sadness"]
        args.out_dir += "_quick"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    tok, model = load_model(args.model_name, args.device, dtype)
    gen_path = out / "generations.csv"

    if not args.score_only:
        queries = json.load(open(args.queries_file))
        fit_q, test_q = queries[:args.n_fit], queries[300:300 + args.n_test]
        prompts = [chat(tok, NEUTRAL, q) for q in test_q]
        layers = sorted(set(itertools.chain(*LAYER_SETS.values())))
        rows = []

        def record(method, config, emotion, responses, **kw):
            for q, r in zip(test_q, responses):
                rows.append(dict(method=method, source=kw.get("source", ""), config=config,
                                 emotion=emotion, query=q, response=r,
                                 layers=kw.get("layers", ""), lam=kw.get("lam"),
                                 aperture=kw.get("aperture"), alpha=kw.get("alpha")))

        def states(source, tone):
            """Fitting states for one tone. model_gen = the model's own generated tokens under
            an emotion system prompt; synthetic/goemotions = teacher-forced dataset replies."""
            if source == "model_gen":
                system = NEUTRAL if tone == "neutral" else EMOTION_PROMPT.format(e=tone)
                return collect_generated_states(tok, model, system, fit_q, layers,
                                                args.fit_tokens, args.batch_size)
            return collect_teacher_forced(tok, model, dataset_items(source, tone, args.n_fit_data),
                                          layers, args.batch_size)

        t0 = time.time()
        base = generate(tok, model, prompts, args.gen_tokens, args.batch_size)
        for e in emotions:
            record("baseline", "baseline", e, base)

        mu_n = {}
        for src in sources:
            print(f"collecting neutral states [{src}]...", flush=True)
            Xn = states(src, "neutral")
            mu_n[src] = {l: Xn[l].mean(0) for l in layers}
            del Xn
        steer = Steerer(model)
        dev = args.device

        for e in emotions:
            for ev_alpha in EV_ALPHAS:
                steer.project_ev(load_project_ev(e, dev), ev_alpha)
                record("project_ev", f"alpha={ev_alpha:g}", e,
                       generate(tok, model, prompts, args.gen_tokens, args.batch_size), alpha=ev_alpha)
                steer.clear()

            for src in sources:
                print(f"[{e}/{src}] collecting states ({time.time() - t0:.0f}s)", flush=True)
                Xe = states(src, e)
                mu_e = {l: Xe[l].mean(0) for l in layers}
                eig = {l: eig_cov(Xe[l]) for l in layers}
                mn = mu_n[src]
                json.dump({"n_samples": int(Xe[layers[0]].shape[0]),
                           "shift_norm": {l: float((mu_e[l] - mn[l]).norm()) for l in layers},
                           "eig_gt_0.5_at_ap0.2": {l: int((eig[l][0] / (eig[l][0] + 25) > 0.5).sum())
                                                   for l in layers}},
                          open(out / f"stats_{e}_{src}.json", "w"), indent=1)
                del Xe

                for ls_name, ls in LAYER_SETS.items():
                    for lam in SHIFT_LAMBDAS:
                        steer.shift_conceptor({l: (mu_e[l].to(dev), mn[l].to(dev), None) for l in ls}, lam)
                        record("shift", f"{src} {ls_name} lam={lam:g}", e,
                               generate(tok, model, prompts, args.gen_tokens, args.batch_size),
                               source=src, layers=ls_name, lam=lam)
                        steer.clear()
                    for a in APERTURES:
                        C = {l: conceptor(*eig[l], a).to(dev) for l in ls}
                        for lam in CONC_LAMBDAS:
                            steer.shift_conceptor({l: (mu_e[l].to(dev), mn[l].to(dev), C[l]) for l in ls}, lam)
                            record("shift_conc", f"{src} {ls_name} ap={a:g} lam={lam:g}", e,
                                   generate(tok, model, prompts, args.gen_tokens, args.batch_size),
                                   source=src, layers=ls_name, lam=lam, aperture=a)
                            steer.clear()
                        del C
                pd.DataFrame(rows).to_csv(gen_path, index=False)   # checkpoint
            print(f"[{e}] done ({time.time() - t0:.0f}s, {len(rows)} generations)", flush=True)

    df = pd.read_csv(gen_path)
    df = score(df, tok, model, args.device, args.batch_size)
    df.to_csv(out / "scored.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.groupby("method")[["conf", "valid_conf", "degen_rate", "ppl_median"]].mean().round(3))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
