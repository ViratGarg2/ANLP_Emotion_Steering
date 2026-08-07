#!/usr/bin/env python3
"""
Capitalisation Conceptor Evaluation for Qwen2.5-1.5B-Instruct

Loads conceptors built by capitalise_extraction.py and evaluates the
capitalisation task: given "word:", does the model predict "Word"?

Steering formula (ref: arXiv:2410.16314, github.com/jorispos/ConceptorSteering):
    h' = beta * C @ (h - mu) + mu

Reports per-layer accuracy (baseline vs steered) and saves a CSV.

Usage:
    python capitalise_evaluation.py --conceptors_dir ./conceptors_cap --beta 3.9
    python capitalise_evaluation.py --conceptors_dir ./conceptors_cap --beta 3.9 --layer 14
    # Sweep multiple betas:
    python capitalise_evaluation.py --conceptors_dir ./conceptors_cap --betas 1.0 2.0 3.9 5.0
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_conceptors(
    conceptors_dir: str,
    aperture: float,
    layers: Optional[List[int]] = None,
) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], dict]:
    """
    Load (C, mu) per layer from conceptors_dir / aperture_X.

    Returns:
        data : Dict[layer -> (C: Tensor(D,D), mu: Tensor(D,))] on CPU
        meta : dict from metadata.json
    """
    cdir = Path(conceptors_dir) / f"aperture_{aperture:.4f}"
    if not cdir.exists():
        # Fallback to older format or exact match
        cdir_fallback = Path(conceptors_dir) / f"aperture_{aperture}"
        if cdir_fallback.exists():
            cdir = cdir_fallback
        else:
            raise FileNotFoundError(f"Cannot find conceptors directory for aperture {aperture}: {cdir}")

    with open(cdir / "metadata.json") as f:
        meta = json.load(f)

    layers_to_load = layers if layers is not None else meta["layers"]
    data = {}
    for l in layers_to_load:
        c_path = cdir / f"layer_{l}_C.pt"
        mu_path = cdir / f"layer_{l}_mu.pt"
        if c_path.exists() and mu_path.exists():
            data[l] = (
                torch.load(c_path, map_location="cpu"),
                torch.load(mu_path, map_location="cpu"),
            )
        else:
            print(f"Warning: missing conceptor files for layer {l}, skipping")

    print(f"Loaded conceptors for {len(data)} layers: {sorted(data.keys())} (aperture={aperture})")
    return data, meta


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CapEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._hooks: List = []

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device.startswith("cuda") else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if not device.startswith("cuda"):
            self.model = self.model.to(device)
        self.model.eval()
        print(f"Loaded: {self.model.config.num_hidden_layers} layers")

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def get_continuation_token_id(self, input_word: str, output_str: str) -> int:
        """
        Return the first token ID the model should predict for output_str
        given that the prompt is "{input_word}:".

        Uses full-string tokenisation so the result respects tokeniser
        context (e.g. byte-pair merges across the colon boundary).
        """
        prompt_ids = self.tokenizer.encode(f"{input_word}:", add_special_tokens=False)
        full_ids = self.tokenizer.encode(f"{input_word}:{output_str}", add_special_tokens=False)
        if len(full_ids) > len(prompt_ids):
            return full_ids[len(prompt_ids)]
        # Fallback: first token of output_str standalone
        ids = self.tokenizer.encode(output_str, add_special_tokens=False)
        return ids[0] if ids else -1

    # ------------------------------------------------------------------
    # Steering hook
    # ------------------------------------------------------------------

    def _install_hook(self, layer: int, C: torch.Tensor, mu: torch.Tensor, beta: float):
        """
        Register a forward hook on model.model.layers[layer] that applies:
            h'_last = beta * C @ (h_last - mu) + mu
        """
        C_dev = C.to(device=self.device, dtype=torch.bfloat16)
        mu_dev = mu.to(device=self.device, dtype=torch.bfloat16)

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h_last = h[:, -1, :]          # (B, D)
            h_mc = h_last - mu_dev        # (B, D)
            # C is symmetric: C @ h_mc == h_mc @ C (per-sample, broadcast over batch)
            h[:, -1, :] = beta * (h_mc @ C_dev) + mu_dev
            return (h,) + out[1:] if isinstance(out, tuple) else h

        handle = self.model.model.layers[layer].register_forward_hook(hook)
        self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next_token_ids(
        self,
        prompts: List[str],
        layer: Optional[int] = None,
        C: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        beta: float = 0.0,
        batch_size: int = 50,
    ) -> torch.Tensor:
        """
        Return argmax next-token IDs (CPU LongTensor, shape=(n_prompts,)).

        Optionally installs conceptor steering on `layer`.
        """
        all_preds = []
        steer = C is not None and beta != 0.0 and layer is not None

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            if steer:
                self._install_hook(layer, C, mu, beta)
            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]   # (B, vocab)
                preds = torch.argmax(logits, dim=-1).cpu()
                all_preds.append(preds)
            finally:
                self._remove_hooks()

        return torch.cat(all_preds)

    def collect_predictions_with_details(
        self,
        eval_pairs: List[dict],
        eval_prompts: List[str],
        expected_ids: torch.Tensor,
        layer: int,
        C: torch.Tensor,
        mu: torch.Tensor,
        beta: float,
        batch_size: int = 50,
        top_k: int = 5,
    ) -> List[dict]:
        """
        Run baseline and steered inference for the best layer and return
        per-sample details including top-k token probabilities.

        Returns a list of dicts — one per sample.
        """
        def _run_with_top_k(prompts, steer: bool):
            all_ids, all_topk_ids, all_topk_probs = [], [], []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                if steer:
                    self._install_hook(layer, C, mu, beta)
                try:
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :].float()   # (B, vocab)
                    probs = torch.softmax(logits, dim=-1)
                    topk = torch.topk(probs, k=top_k, dim=-1)
                    all_ids.append(torch.argmax(probs, dim=-1).cpu())
                    all_topk_ids.append(topk.indices.cpu())
                    all_topk_probs.append(topk.values.cpu())
                finally:
                    self._remove_hooks()
            return (
                torch.cat(all_ids),
                torch.cat(all_topk_ids, dim=0),
                torch.cat(all_topk_probs, dim=0),
            )

        print(f"Collecting detailed outputs for layer {layer}, beta={beta}...")
        base_ids, base_topk_ids, base_topk_probs = _run_with_top_k(eval_prompts, steer=False)
        steer_ids, steer_topk_ids, steer_topk_probs = _run_with_top_k(eval_prompts, steer=True)

        records = []
        for i, pair in enumerate(eval_pairs):
            exp_id = int(expected_ids[i].item())
            b_id = int(base_ids[i].item())
            s_id = int(steer_ids[i].item())

            records.append({
                "input_word":       pair["input"],
                "prompt":           eval_prompts[i],
                "expected_output":  pair["output"],
                "expected_token":   self.tokenizer.decode([exp_id]),
                "baseline_token":   self.tokenizer.decode([b_id]),
                "baseline_correct": b_id == exp_id,
                "steered_token":    self.tokenizer.decode([s_id]),
                "steered_correct":  s_id == exp_id,
                "layer":            layer,
                "beta":             beta,
                "baseline_top5": [
                    {
                        "token": self.tokenizer.decode([int(base_topk_ids[i, k].item())]),
                        "token_id": int(base_topk_ids[i, k].item()),
                        "prob": round(float(base_topk_probs[i, k].item()), 6),
                    }
                    for k in range(top_k)
                ],
                "steered_top5": [
                    {
                        "token": self.tokenizer.decode([int(steer_topk_ids[i, k].item())]),
                        "token_id": int(steer_topk_ids[i, k].item()),
                        "prob": round(float(steer_topk_probs[i, k].item()), 6),
                    }
                    for k in range(top_k)
                ],
            })

        return records

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_pairs,
        conceptor_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        beta: float,
        n_eval: int = 1000,
        batch_size: int = 50,
    ) -> dict:
        """
        Evaluate capitalisation accuracy per layer.

        Returns dict with:
            baseline_accuracy, beta, n_eval, layers: {layer: {accuracy, improvement}},
            _eval_pairs, _eval_prompts, _expected_ids  ← kept for detailed output generation
        """
        # Cap test set
        if n_eval and len(test_pairs) > n_eval:
            eval_pairs = random.sample(test_pairs, n_eval)
        else:
            eval_pairs = list(test_pairs)

        # Build prompts and expected first-token IDs
        eval_prompts = [f"{p['input']}:" for p in eval_pairs]
        expected_ids = torch.tensor(
            [self.get_continuation_token_id(p["input"], p["output"]) for p in eval_pairs],
            dtype=torch.long,
        )

        # Drop pairs where tokenisation failed
        valid = expected_ids != -1
        eval_pairs    = [p for p, v in zip(eval_pairs,   valid.tolist()) if v]
        eval_prompts  = [p for p, v in zip(eval_prompts, valid.tolist()) if v]
        expected_ids  = expected_ids[valid]
        n_invalid = int((~valid).sum().item())
        if n_invalid:
            print(f"Dropped {n_invalid} pairs with invalid tokenisation")
        print(f"Evaluating {len(eval_prompts)} prompts")

        # ---- Baseline ------------------------------------------------
        print("Running baseline (no steering)...")
        baseline_preds = self.predict_next_token_ids(eval_prompts, batch_size=batch_size)
        baseline_acc = (baseline_preds == expected_ids).float().mean().item() * 100
        print(f"Baseline accuracy: {baseline_acc:.2f}%")

        # ---- Per-layer steered evaluation ----------------------------
        results: dict = {
            "baseline_accuracy": baseline_acc,
            "beta": beta,
            "n_eval": len(eval_prompts),
            "layers": {},
            # Kept for detailed output generation in main(); not written to CSV
            "_eval_pairs":    eval_pairs,
            "_eval_prompts":  eval_prompts,
            "_expected_ids":  expected_ids,
        }

        for layer, (C, mu) in tqdm(sorted(conceptor_data.items()), desc="Layers"):
            preds = self.predict_next_token_ids(
                eval_prompts,
                layer=layer,
                C=C,
                mu=mu,
                beta=beta,
                batch_size=batch_size,
            )
            acc = (preds == expected_ids).float().mean().item() * 100
            delta = acc - baseline_acc
            results["layers"][layer] = {"accuracy": acc, "improvement": delta}
            print(f"  Layer {layer:2d}: {acc:.2f}%  ({delta:+.2f}% vs baseline)")

        return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_csv(results_list: List[dict], meta: dict, output_file: str):
    """Save a list of per-beta result dicts to CSV."""
    rows = []
    for res in results_list:
        beta = res["beta"]
        baseline = res["baseline_accuracy"]
        for layer, lr in sorted(res["layers"].items()):
            rows.append({
                "layer": layer,
                "beta": beta,
                "aperture": res.get("aperture", meta.get("aperture", "?")),
                "n_eval": res["n_eval"],
                "baseline_accuracy": f"{baseline:.2f}",
                "steered_accuracy": f"{lr['accuracy']:.2f}",
                "improvement": f"{lr['improvement']:+.2f}",
                "model": meta.get("model_name", "?"),
            })

    if not rows:
        print("No results to save.")
        return

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {output_file}")


def save_detailed_outputs(records: List[dict], output_file: str):
    """
    Save per-sample baseline vs steered predictions to a JSON file.

    Each record contains:
        input_word, prompt, expected_output, expected_token,
        baseline_token, baseline_correct,
        steered_token,  steered_correct,
        layer, beta,
        baseline_top5: [{token, token_id, prob}, ...],
        steered_top5:  [{token, token_id, prob}, ...]
    """
    n_correct_base  = sum(1 for r in records if r["baseline_correct"])
    n_correct_steer = sum(1 for r in records if r["steered_correct"])
    n = len(records)
    summary = {
        "n_samples":          n,
        "baseline_accuracy":  round(n_correct_base  / n * 100, 2),
        "steered_accuracy":   round(n_correct_steer / n * 100, 2),
        "improvement":        round((n_correct_steer - n_correct_base) / n * 100, 2),
        "best_layer":         records[0]["layer"] if records else None,
        "beta":               records[0]["beta"]  if records else None,
    }
    payload = {"summary": summary, "samples": records}
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"Detailed outputs saved to {output_file}  "
        f"({summary['steered_accuracy']:.2f}% steered vs "
        f"{summary['baseline_accuracy']:.2f}% baseline)"
    )


def print_summary(results_list: List[dict]):
    for res in results_list:
        beta = res["beta"]
        aperture = res.get("aperture", "?")
        baseline = res["baseline_accuracy"]
        if not res["layers"]:
            continue
        best_layer = max(res["layers"], key=lambda l: res["layers"][l]["accuracy"])
        best_acc = res["layers"][best_layer]["accuracy"]
        best_delta = res["layers"][best_layer]["improvement"]
        print(
            f"  aperture={aperture} | beta={beta:.1f} | best layer={best_layer} | "
            f"{best_acc:.2f}% steered (+{best_delta:+.2f}% over {baseline:.2f}% baseline)"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate capitalisation conceptor steering on Qwen2.5-1.5B-Instruct",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--conceptors_dir", type=str, default="./conceptors_cap")
    parser.add_argument("--apertures", type=float, nargs="+", default=[0.001, 0.0125, 0.05, 0.1,0.2,0.5, 1.0,2.0,5.0,10.0],
                        help="List of apertures to evaluate.")
    parser.add_argument("--beta", type=float, default=3.9,
                        help="Single beta value (ignored if --betas is set)")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5,1.0,2.0,3.0,5.0],
                        help="Sweep multiple beta values, e.g. --betas 1.0 2.0 3.9 5.0")
    parser.add_argument("--n_eval", type=int, default=0,
                        help="Max test samples per beta (0 = use all test pairs)")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Override layers to evaluate (default: all saved layers)")
    parser.add_argument("--test_pairs", type=str, default=None,
                        help="Path to test_pairs.json (default: auto from conceptors_dir)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        args.device = "cpu"

    random.seed(args.seed)

    # We defer conceptor loading into the aperture loop
    master_meta = None
    loaded_conceptors_by_ap = {}

    test_path = args.test_pairs or str(Path(args.conceptors_dir) / "test_pairs.json")
    with open(test_path) as f:
        test_pairs = json.load(f)
    print(f"Loaded {len(test_pairs)} test pairs from {test_path}")

    betas = args.betas if args.betas else [args.beta]
    n_eval = args.n_eval if args.n_eval > 0 else len(test_pairs)

    evaluator = CapEvaluator(model_name=args.model_name, device=args.device)

    all_results = []
    
    for ap in args.apertures:
        print(f"\n{'='*60}")
        print(f" Evaluating aperture = {ap}")
        print(f"{'='*60}")
        
        try:
            conceptor_data, meta = load_conceptors(args.conceptors_dir, ap, args.layers)
            loaded_conceptors_by_ap[ap] = conceptor_data
            if master_meta is None:
                master_meta = meta
        except FileNotFoundError as e:
            print(e)
            continue

        for beta in betas:
            print(f"\n  -- beta = {beta} --")
            res = evaluator.evaluate(
                test_pairs=test_pairs,
                conceptor_data=conceptor_data,
                beta=beta,
                n_eval=n_eval,
                batch_size=args.batch_size,
            )
            res["aperture"] = ap
            all_results.append(res)

    if not all_results:
        print("No results to save.")
        return

    # Save per-layer accuracy CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    beta_str = "_".join(str(b) for b in betas)
    output_file = args.output_file or f"cap_results_beta{beta_str}_multiap_{ts}.csv"
    save_csv(all_results, master_meta or {}, output_file)

    print("\n--- Summary ---")
    print_summary(all_results)

    # ------------------------------------------------------------------
    # Detailed per-sample outputs for the best (aperture, beta, layer) configuration
    # ------------------------------------------------------------------
    best_res = max(
        all_results,
        key=lambda r: max((v["accuracy"] for v in r["layers"].values()), default=0.0),
    )
    best_layer = max(best_res["layers"], key=lambda l: best_res["layers"][l]["accuracy"])
    best_beta  = best_res["beta"]
    best_aperture = best_res["aperture"]
    
    C_best, mu_best = loaded_conceptors_by_ap[best_aperture][best_layer]

    print(f"\nGenerating detailed outputs for best config: aperture={best_aperture}, layer={best_layer}, beta={best_beta}")
    records = evaluator.collect_predictions_with_details(
        eval_pairs    = best_res["_eval_pairs"],
        eval_prompts  = best_res["_eval_prompts"],
        expected_ids  = best_res["_expected_ids"],
        layer         = best_layer,
        C             = C_best,
        mu            = mu_best,
        beta          = best_beta,
        batch_size    = args.batch_size,
        top_k         = 5,
    )
    details_file = f"cap_details_ap{best_aperture}_layer{best_layer}_beta{best_beta}_{ts}.json"
    save_detailed_outputs(records, details_file)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
