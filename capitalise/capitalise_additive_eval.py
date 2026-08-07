#!/usr/bin/env python3
"""
Capitalisation Additive Steering Evaluation for Qwen2.5-1.5B-Instruct

Additive (function-vector) baseline for direct comparison with the
conceptor-based approach in capitalise_evaluation.py.

Steering formula:
    h' = h + beta * mu_layer
where mu_layer = mean(X_task) at layer l, saved by capitalise_extraction.py.
This is the mean-centred task activation vector — equivalent to the
"function vector" from the FunctionVectors paper.

Reference:
    Todd et al. (2023) "Function Vectors in Large Language Models"
    Notebook: github.com/jorispos/ConceptorSteering (additive baseline)

Usage:
    python capitalise_additive_eval.py --conceptors_dir ./conceptors_cap --beta 2.3
    python capitalise_additive_eval.py --conceptors_dir ./conceptors_cap --betas 0.5 1.0 2.3 5.0
"""

import argparse
import csv
import json
import random
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

def load_steering_vectors(
    conceptors_dir: str,
    layers: Optional[List[int]] = None,
) -> Tuple[Dict[int, torch.Tensor], dict]:
    """
    Load per-layer task-mean vectors (mu) saved by capitalise_extraction.py.

    Returns:
        vectors : Dict[layer -> mu Tensor(D,)] on CPU
        meta    : dict from metadata.json
    """
    cdir = Path(conceptors_dir)
    with open(cdir / "metadata.json") as f:
        meta = json.load(f)

    layers_to_load = layers if layers is not None else meta["layers"]
    vectors: Dict[int, torch.Tensor] = {}
    for l in layers_to_load:
        mu_path = cdir / f"layer_{l}_mu.pt"
        if mu_path.exists():
            vectors[l] = torch.load(mu_path, map_location="cpu")
        else:
            print(f"Warning: missing mu file for layer {l}, skipping")

    print(f"Loaded steering vectors for {len(vectors)} layers: {sorted(vectors.keys())}")
    return vectors, meta


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class AdditiveCapEvaluator:
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
        """First token ID the model should predict for output_str after '{input_word}:'."""
        prompt_ids = self.tokenizer.encode(f"{input_word}:", add_special_tokens=False)
        full_ids   = self.tokenizer.encode(f"{input_word}:{output_str}", add_special_tokens=False)
        if len(full_ids) > len(prompt_ids):
            return full_ids[len(prompt_ids)]
        ids = self.tokenizer.encode(output_str, add_special_tokens=False)
        return ids[0] if ids else -1

    # ------------------------------------------------------------------
    # Steering hook  —  h' = h + beta * mu
    # ------------------------------------------------------------------

    def _install_hook(self, layer: int, mu: torch.Tensor, beta: float):
        """Add scaled task-mean vector to the last token's hidden state."""
        sv = (mu * beta).to(device=self.device, dtype=torch.bfloat16)   # pre-scale once

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, -1, :] += sv
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
        mu: Optional[torch.Tensor] = None,
        beta: float = 0.0,
        batch_size: int = 50,
    ) -> torch.Tensor:
        """Return argmax next-token IDs (CPU LongTensor). Optionally steer."""
        all_preds = []
        steer = mu is not None and beta != 0.0 and layer is not None

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            if steer:
                self._install_hook(layer, mu, beta)
            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                outputs = self.model(**inputs)
                logits  = outputs.logits[:, -1, :]
                all_preds.append(torch.argmax(logits, dim=-1).cpu())
            finally:
                self._remove_hooks()

        return torch.cat(all_preds)

    def collect_predictions_with_details(
        self,
        eval_pairs: List[dict],
        eval_prompts: List[str],
        expected_ids: torch.Tensor,
        layer: int,
        mu: torch.Tensor,
        beta: float,
        batch_size: int = 50,
        top_k: int = 5,
    ) -> List[dict]:
        """
        Run baseline and steered inference; return per-sample dicts with
        top-k token probabilities for both passes.
        """
        def _run(prompts, steer: bool):
            all_ids, all_topk_ids, all_topk_probs = [], [], []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                if steer:
                    self._install_hook(layer, mu, beta)
                try:
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    outputs = self.model(**inputs)
                    logits  = outputs.logits[:, -1, :].float()
                    probs   = torch.softmax(logits, dim=-1)
                    topk    = torch.topk(probs, k=top_k, dim=-1)
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
        base_ids,  base_topk_ids,  base_topk_probs  = _run(eval_prompts, steer=False)
        steer_ids, steer_topk_ids, steer_topk_probs = _run(eval_prompts, steer=True)

        records = []
        for i, pair in enumerate(eval_pairs):
            exp_id = int(expected_ids[i].item())
            b_id   = int(base_ids[i].item())
            s_id   = int(steer_ids[i].item())
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
                        "token":    self.tokenizer.decode([int(base_topk_ids[i, k].item())]),
                        "token_id": int(base_topk_ids[i, k].item()),
                        "prob":     round(float(base_topk_probs[i, k].item()), 6),
                    }
                    for k in range(top_k)
                ],
                "steered_top5": [
                    {
                        "token":    self.tokenizer.decode([int(steer_topk_ids[i, k].item())]),
                        "token_id": int(steer_topk_ids[i, k].item()),
                        "prob":     round(float(steer_topk_probs[i, k].item()), 6),
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
        steering_vectors: Dict[int, torch.Tensor],
        beta: float,
        n_eval: int = 0,
        batch_size: int = 50,
    ) -> dict:
        """
        Evaluate additive steering accuracy per layer.

        Returns dict with baseline_accuracy, per-layer accuracy/improvement,
        plus _eval_pairs/_eval_prompts/_expected_ids for detailed output.
        """
        if n_eval and len(test_pairs) > n_eval:
            eval_pairs = random.sample(test_pairs, n_eval)
        else:
            eval_pairs = list(test_pairs)

        eval_prompts = [f"{p['input']}:" for p in eval_pairs]
        expected_ids = torch.tensor(
            [self.get_continuation_token_id(p["input"], p["output"]) for p in eval_pairs],
            dtype=torch.long,
        )

        valid         = expected_ids != -1
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
        baseline_acc   = (baseline_preds == expected_ids).float().mean().item() * 100
        print(f"Baseline accuracy: {baseline_acc:.2f}%")

        # ---- Per-layer additive steering evaluation ------------------
        results: dict = {
            "baseline_accuracy": baseline_acc,
            "beta":              beta,
            "n_eval":            len(eval_prompts),
            "layers":            {},
            "_eval_pairs":       eval_pairs,
            "_eval_prompts":     eval_prompts,
            "_expected_ids":     expected_ids,
        }

        for layer, mu in tqdm(sorted(steering_vectors.items()), desc="Layers"):
            preds = self.predict_next_token_ids(
                eval_prompts, layer=layer, mu=mu, beta=beta, batch_size=batch_size
            )
            acc   = (preds == expected_ids).float().mean().item() * 100
            delta = acc - baseline_acc
            results["layers"][layer] = {"accuracy": acc, "improvement": delta}
            print(f"  Layer {layer:2d}: {acc:.2f}%  ({delta:+.2f}% vs baseline)")

        return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_csv(results_list: List[dict], meta: dict, output_file: str):
    rows = []
    for res in results_list:
        beta     = res["beta"]
        baseline = res["baseline_accuracy"]
        for layer, lr in sorted(res["layers"].items()):
            rows.append({
                "layer":             layer,
                "beta":              beta,
                "aperture":          meta.get("aperture", "?"),
                "n_eval":            res["n_eval"],
                "baseline_accuracy": f"{baseline:.2f}",
                "steered_accuracy":  f"{lr['accuracy']:.2f}",
                "improvement":       f"{lr['improvement']:+.2f}",
                "method":            "additive",
                "model":             meta.get("model_name", "?"),
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
    """Save per-sample baseline vs steered predictions to JSON."""
    n_correct_base  = sum(1 for r in records if r["baseline_correct"])
    n_correct_steer = sum(1 for r in records if r["steered_correct"])
    n = len(records)
    summary = {
        "method":             "additive",
        "n_samples":          n,
        "baseline_accuracy":  round(n_correct_base  / n * 100, 2),
        "steered_accuracy":   round(n_correct_steer / n * 100, 2),
        "improvement":        round((n_correct_steer - n_correct_base) / n * 100, 2),
        "best_layer":         records[0]["layer"] if records else None,
        "beta":               records[0]["beta"]  if records else None,
    }
    with open(output_file, "w") as f:
        json.dump({"summary": summary, "samples": records}, f, indent=2)
    print(
        f"Detailed outputs saved to {output_file}  "
        f"({summary['steered_accuracy']:.2f}% steered vs "
        f"{summary['baseline_accuracy']:.2f}% baseline)"
    )


def print_summary(results_list: List[dict]):
    for res in results_list:
        if not res["layers"]:
            continue
        best_layer = max(res["layers"], key=lambda l: res["layers"][l]["accuracy"])
        best_acc   = res["layers"][best_layer]["accuracy"]
        best_delta = res["layers"][best_layer]["improvement"]
        print(
            f"  beta={res['beta']:.1f} | best layer={best_layer} | "
            f"{best_acc:.2f}% steered ({best_delta:+.2f}% over {res['baseline_accuracy']:.2f}% baseline)"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate additive (function-vector) steering for capitalisation on Qwen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--conceptors_dir", type=str, default="./conceptors_cap",
                        help="Directory produced by capitalise_extraction.py (contains mu .pt files)")
    parser.add_argument("--beta", type=float, default=2.3,
                        help="Single beta value (ignored if --betas is set)")
    parser.add_argument("--betas", type=float, nargs="+", default=[-1.0, 0.5, 1.0, 2.5, 3.0,4.0,5.0],
                        help="Sweep multiple betas, e.g. --betas 0.5 1.0 2.5 3.0")
    parser.add_argument("--n_eval", type=int, default=0,
                        help="Max test samples (0 = all)")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Override layers (default: all saved layers)")
    parser.add_argument("--test_pairs", type=str, default=None,
                        help="Path to test_pairs.json (default: auto from conceptors_dir)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        args.device = "cpu"

    random.seed(args.seed)

    # Load steering vectors (mu per layer)
    steering_vectors, meta = load_steering_vectors(args.conceptors_dir, args.layers)

    # Load test pairs
    test_path = args.test_pairs or str(Path(args.conceptors_dir) / "test_pairs.json")
    with open(test_path) as f:
        test_pairs = json.load(f)
    print(f"Loaded {len(test_pairs)} test pairs from {test_path}")

    betas  = args.betas if args.betas else [args.beta]
    n_eval = args.n_eval if args.n_eval > 0 else len(test_pairs)

    evaluator = AdditiveCapEvaluator(model_name=args.model_name, device=args.device)

    all_results = []
    for beta in betas:
        print(f"\n{'='*60}")
        print(f"  Additive steering  |  beta = {beta}")
        print(f"{'='*60}")
        res = evaluator.evaluate(
            test_pairs=test_pairs,
            steering_vectors=steering_vectors,
            beta=beta,
            n_eval=n_eval,
            batch_size=args.batch_size,
        )
        all_results.append(res)

    # Save per-layer accuracy CSV
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    aperture  = meta.get("aperture", "x")
    beta_str  = "_".join(str(b) for b in betas)
    output_file = args.output_file or f"add_results_beta{beta_str}_ap{aperture}_{ts}.csv"
    save_csv(all_results, meta, output_file)

    print("\n--- Summary ---")
    print_summary(all_results)

    # ------------------------------------------------------------------
    # Detailed per-sample outputs for the best (beta, layer) combination
    # ------------------------------------------------------------------
    best_res   = max(
        all_results,
        key=lambda r: max((v["accuracy"] for v in r["layers"].values()), default=0.0),
    )
    best_layer = max(best_res["layers"], key=lambda l: best_res["layers"][l]["accuracy"])
    best_beta  = best_res["beta"]
    mu_best    = steering_vectors[best_layer]

    print(f"\nGenerating detailed outputs for best config: layer={best_layer}, beta={best_beta}")
    records = evaluator.collect_predictions_with_details(
        eval_pairs   = best_res["_eval_pairs"],
        eval_prompts = best_res["_eval_prompts"],
        expected_ids = best_res["_expected_ids"],
        layer        = best_layer,
        mu           = mu_best,
        beta         = best_beta,
        batch_size   = args.batch_size,
        top_k        = 5,
    )
    details_file = f"add_details_layer{best_layer}_beta{best_beta}_ap{aperture}_{ts}.json"
    save_detailed_outputs(records, details_file)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
