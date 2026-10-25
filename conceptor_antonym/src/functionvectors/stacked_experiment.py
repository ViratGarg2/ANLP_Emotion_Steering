"""
Stacked shift+conceptor on the single-token function tasks, benchmarked against the published
single-layer methods on the SAME prompt draw (same data generation and scoring as
run_experiment.py; seed 0, 100 ICL prompts x 10 examples, zero-shot "x:" test inputs).

Methods (all applied at blocks.l.hook_resid_pre, last token, one forward pass):
    addition    h + beta * mean(X_task)                     single layers (Todd et al. / ActAdd)
    paper       beta * C(R_task, alpha) h                   single layers (Postmus & Abreu, Eq. 4)
    shift       h + lam * (mu_task - mu_zero)               single layers, blocks, all layers
    shift_conc  h + lam * [(mu_task - mu_zero) + (C_task - I)(h - mu_zero)]
                                                            single layers, blocks, all layers
X_task: last-token activations of the ICL prompts "a:A, b:B, ..., x:"; X_zero: last-token
activations of the matching zero-shot prompt "x:" (the same final input without demonstrations),
so mu_task - mu_zero isolates what the demonstrations add. C_task = C(cov(X_task), alpha).

    python stacked_experiment.py --task antonyms --results_path ../../Results_stacked
    python stacked_experiment.py --task capitalize --quick      # smoke test
"""

import argparse
import math
import os
import random
import time

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from utils.data import generate_experiment_data
from utils.steering import extract_activations_last_token, top_1_first_tokens

ADD_BETAS = [0.5, 1.0, 2.0, 2.3, 3.0, 4.0, 5.0]               # as in the earlier sweeps
PAPER_BETAS, PAPER_APS = [0.5, 1.0, 2.0, 3.9, 5.0], [0.005, 0.05, 0.1, 1.0, 2.0]
SHIFT_LAMS = [0.25, 0.5, 1.0, 1.5, 2.0]
CONC_APS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
CONC_LAMS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def conceptor_from(M, aperture):
    s, U = torch.linalg.eigh(M.double())
    s = s.clamp(min=0)
    return ((U * (s / (s + aperture ** -2))) @ U.T).float()


def accuracy(model, prompts, answers, hooks, batch_size):
    hits = 0
    for i in range(0, len(prompts), batch_size):
        top = top_1_first_tokens(model, prompts[i:i + batch_size], fwd_hooks=hooks)
        hits += sum(t == a for t, a in zip(top, answers[i:i + batch_size]))
    return hits / len(prompts)


def last_token_hook(fn):
    def hook(resid_pre, hook):
        resid_pre[:, -1, :] = fn(resid_pre[:, -1, :].float()).to(resid_pre.dtype)
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--num_steering_prompts", type=int, default=100)
    ap.add_argument("--num_steering_examples_per_prompt", type=int, default=10)
    ap.add_argument("--num_input_prompts", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results_path", default="../../Results_stacked")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = HookedTransformer.from_pretrained_no_processing(model_name=args.model_name, device=device,
                                                            dtype=args.dtype)
    model.eval()
    n_layers = model.cfg.n_layers
    data = generate_experiment_data(model=model, num_experiments=1,
                                    steering_prompts_path=f"../../data/functionvectors/{args.task}.json",
                                    num_steering_prompts=args.num_steering_prompts,
                                    num_steering_examples_per_prompt=args.num_steering_examples_per_prompt,
                                    num_input_prompts=args.num_input_prompts)[0]
    icl = data["steering_examples"]
    zero = [p.rsplit(", ", 1)[-1] for p in icl]          # "a:A, ..., x:"  ->  "x:"
    test, answers = data["prompts_to_steer"], data["correct_outputs_1st"]

    singles = {f"L{l}": [l] for l in range(n_layers)}
    blocks = {f"L{a}-{a + 6}": list(range(a, a + 7)) for a in range(4, n_layers - 6, 2)}
    blocks |= {"L8-20": list(range(8, 21)), f"L0-{n_layers - 1}": list(range(n_layers))}
    global ADD_BETAS, PAPER_BETAS, PAPER_APS, SHIFT_LAMS, CONC_APS, CONC_LAMS
    if args.quick:
        test, answers = test[:64], answers[:64]
        singles = {k: v for k, v in singles.items() if k in ("L14", "L20")}
        blocks = {k: v for k, v in blocks.items() if k in ("L14-20", f"L0-{n_layers - 1}")}
        ADD_BETAS, PAPER_BETAS, PAPER_APS, SHIFT_LAMS = [2.0], [1.0], [0.1], [1.0]
        CONC_APS, CONC_LAMS = [1.0], [0.1]

    layers = list(range(n_layers))
    t0 = time.time()
    Xt = extract_activations_last_token(model, icl, layers, device)
    Xz = extract_activations_last_token(model, zero, layers, device)
    mu_t = {l: Xt[l].float().mean(0) for l in layers}
    mu_z = {l: Xz[l].float().mean(0) for l in layers}
    cov_eig = {}
    print(f"activations cached ({time.time() - t0:.0f}s): {Xt[0].shape[0]} ICL rows, d={Xt[0].shape[1]}", flush=True)

    rows = []

    def run(method, layer_set, ls, params, fns):
        hooks = [(f"blocks.{l}.hook_resid_pre", last_token_hook(fns[l])) for l in ls]
        acc = accuracy(model, test, answers, hooks, args.batch_size)
        rows.append({"task": args.task, "method": method, "layers": layer_set, "n_layers": len(ls), **params,
                     "accuracy": acc})

    base = accuracy(model, test, answers, [], args.batch_size)
    rows.append({"task": args.task, "method": "baseline", "layers": "", "n_layers": 0, "accuracy": base})
    print(f"baseline accuracy {base:.3f}", flush=True)

    dev = lambda t: t.to(device)
    for name, ls in singles.items():                       # published single-layer methods
        l = ls[0]
        v = dev(mu_t[l])
        for b in ADD_BETAS:
            run("addition", name, ls, {"beta": b}, {l: (lambda h, v=v, b=b: h + b * v)})
        R = (Xt[l].float().T @ Xt[l].float()) / Xt[l].shape[0]
        for a in PAPER_APS:
            C = dev(conceptor_from(R, a))
            for b in PAPER_BETAS:
                run("paper", name, ls, {"beta": b, "aperture": a}, {l: (lambda h, C=C, b=b: b * (h @ C))})
        print(f"{name} single-layer published methods done ({time.time() - t0:.0f}s)", flush=True)

    for name, ls in {**singles, **blocks}.items():          # ours: shift and shift+conceptor
        shift = {l: dev(mu_t[l] - mu_z[l]) for l in ls}
        mz = {l: dev(mu_z[l]) for l in ls}
        for lam in SHIFT_LAMS:
            run("shift", name, ls, {"lam": lam}, {l: (lambda h, s=shift[l], lam=lam: h + lam * s) for l in ls})
        for l in ls:
            if l not in cov_eig:
                X = Xt[l].float()
                Xc = X - X.mean(0)
                cov_eig[l] = (Xc.T @ Xc) / X.shape[0]
        for a in CONC_APS:
            C = {l: dev(conceptor_from(cov_eig[l], a)) for l in ls}
            for lam in CONC_LAMS:
                fns = {l: (lambda h, s=shift[l], m=mz[l], C=C[l], lam=lam:
                           h + lam * (s + (h - m) @ C - (h - m))) for l in ls}
                run("shift_conc", name, ls, {"lam": lam, "aperture": a}, fns)
        print(f"{name} shift / shift_conc done ({time.time() - t0:.0f}s, {len(rows)} configs)", flush=True)
        pd.DataFrame(rows).to_csv(out_file(args), index=False)    # checkpoint

    df = pd.DataFrame(rows)
    df.to_csv(out_file(args), index=False)
    best = df.loc[df.groupby(["method", "n_layers"]).accuracy.idxmax()] if len(df) else df
    print(best.sort_values("accuracy", ascending=False).to_string(index=False))
    print(f"wrote {out_file(args)}")


def out_file(args):
    os.makedirs(args.results_path, exist_ok=True)
    return f"{args.results_path}/Qwen2.5-1.5B-Instruct_{args.task}_stacked{'_quick' if args.quick else ''}.csv"


if __name__ == "__main__":
    main()
