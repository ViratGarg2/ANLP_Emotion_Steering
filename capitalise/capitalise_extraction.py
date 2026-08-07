#!/usr/bin/env python3
"""
Capitalisation Conceptor Extraction for Qwen2.5-1.5B-Instruct

Builds conceptors for the capitalisation function-vector task:
1. Creates few-shot prompts from capitalize.json (e.g. "cat:Cat, dog:Dog, bird:")
2. Extracts last-token hidden states at each transformer layer
3. Mean-centres by task mean and computes C(R, aperture) = R(R + aperture^-2 I)^-1
4. Saves per-layer C and mu as .pt files, plus a held-out test split

Reference: arXiv:2410.16314 / github.com/jorispos/ConceptorSteering

Usage:
    python capitalise_extraction.py
    python capitalise_extraction.py --aperture 0.05 --n_train 100 --layers 9 10 11 12 13 14 15 16 17
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Conceptor maths
# ---------------------------------------------------------------------------

def compute_conceptor(X: torch.Tensor, aperture: float) -> torch.Tensor:
    """
    C(R, aperture) = R (R + aperture^-2 I)^-1  via SVD of R = X.T X / n.

    X : (n, hidden_dim) float32 on CPU
    Returns symmetric (hidden_dim, hidden_dim) float32 with eigenvalues in [0, 1].
    """
    X = X.float()
    R = torch.matmul(X.T, X) / X.shape[0]
    # torch.linalg.svd on symmetric PSD is equivalent to eigh but more numerically stable
    U, S, _ = torch.linalg.svd(R, full_matrices=False)
    scale = S / (S + aperture ** (-2))
    return (U * scale @ U.T).float()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_pairs(path: str):
    with open(path, "r") as f:
        return json.load(f)


def create_few_shot_prompt(pairs, n_examples: int) -> str:
    """
    Returns a few-shot capitalisation prompt like:
        "cat:Cat, dog:Dog, ..., bird:"
    The last pair is missing its output (the model should predict it).
    """
    sampled = random.sample(pairs, n_examples)
    parts = [f"{p['input']}:{p['output']}" for p in sampled[:-1]]
    parts.append(f"{sampled[-1]['input']}:")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class CapConceptorExtractor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        self.model_name = model_name
        self.device = device

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

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        print(f"Loaded: {self.num_layers} layers, hidden_size={self.hidden_size}")

    def extract_activations(
        self,
        prompts,
        layers,
        batch_size: int = 50,
        desc: str = "Extracting",
    ) -> dict:
        """
        Run forward passes and capture last-token hidden states at each layer.

        Returns Dict[layer -> Tensor(n_prompts, hidden_dim)] float32 on CPU.
        Hook point: output of model.model.layers[l] (= hook_resid_post[l]).
        """
        storage = {l: [] for l in layers}
        hooks = []

        def make_hook(l):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                # Capture last-token (works for left-padded batches)
                storage[l].append(h[:, -1, :].detach().cpu().float())
            return hook

        for l in layers:
            hooks.append(self.model.model.layers[l].register_forward_hook(make_hook(l)))

        try:
            for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
                batch = prompts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        return {l: torch.cat(tensors, dim=0) for l, tensors in storage.items() if tensors}

    def build_and_save(
        self,
        pairs,
        layers,
        apertures: list[float],
        n_train: int,
        n_examples: int,
        output_dir: str,
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Hold out 20 % for evaluation (at least 100 pairs)
        random.shuffle(pairs)
        n_test = max(100, len(pairs) // 5)
        test_pairs = pairs[:n_test]
        train_pairs = pairs[n_test:]

        with open(output_path / "test_pairs.json", "w") as f:
            json.dump(test_pairs, f, indent=2)
        print(f"Train: {len(train_pairs)} pairs  |  Test: {len(test_pairs)} pairs")

        # Create few-shot training prompts
        print(f"Creating {n_train} few-shot prompts ({n_examples} in-context examples each)...")
        prompts = [create_few_shot_prompt(train_pairs, n_examples) for _ in range(n_train)]

        # Extract last-token activations at every requested layer
        activations = self.extract_activations(prompts, layers, desc="Extracting task activations")

        # Build mean-centred conceptors
        print("Computing mean-centred conceptors for multiple apertures...")
        for l, X in tqdm(sorted(activations.items()), desc="Building conceptors"):
            mu = X.mean(dim=0)            # (D,)  task training mean
            X_mc = X-mu                 # (n, D) mean-centred activations

            for ap in apertures:
                C = compute_conceptor(X_mc, ap)   # (D, D)

                ap_dir = output_path / f"aperture_{ap:.4f}"
                ap_dir.mkdir(parents=True, exist_ok=True)

                torch.save(C, ap_dir / f"layer_{l}_C.pt")
                torch.save(mu, ap_dir / f"layer_{l}_mu.pt")

                if l == layers[0]:
                    metadata = {
                        "model_name": self.model_name,
                        "aperture": ap,
                        "n_train": n_train,
                        "n_examples_per_prompt": n_examples,
                        "layers": layers,
                        "hidden_size": self.hidden_size,
                        "num_layers": self.num_layers,
                        "n_test_pairs": len(test_pairs),
                        "timestamp": datetime.now().isoformat(),
                        "method": "mean_centred_conceptor",
                        "hook_point": "register_forward_hook on model.model.layers[l] (resid_post)",
                        "steering_formula": "h' = beta * C @ (h - mu) + mu",
                        "reference": "arXiv:2410.16314, github.com/jorispos/ConceptorSteering",
                    }
                    with open(ap_dir / "metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)

        print(f"\nConceptors saved to {output_dir}")
        print(f"Next: python capitalise_evaluation.py --conceptors_dir {output_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract capitalisation conceptors from Qwen2.5-1.5B-Instruct",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--capitalize_json", type=str, default="capitalize.json")
    parser.add_argument("--apertures", type=float, nargs="+", default=[0.001, 0.0125, 0.05, 0.1,0.2,0.5, 1.0, 2.0,5.0,10.0],
                        help="List of aperture α values. C(R,α). Conceptors will be saved in subdirectories.")
    parser.add_argument("--n_train", type=int, default=100,
                        help="Number of few-shot prompts for conceptor extraction")
    parser.add_argument("--n_examples", type=int, default=10,
                        help="In-context examples per few-shot prompt")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(1, 25)),
                        help="Transformer layer indices to extract from (0-indexed)")
    parser.add_argument("--output_dir", type=str, default="./conceptors_cap")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        args.device = "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = load_pairs(args.capitalize_json)
    print(f"Loaded {len(pairs)} pairs from {args.capitalize_json}")

    extractor = CapConceptorExtractor(model_name=args.model_name, device=args.device)
    extractor.build_and_save(
        pairs=pairs,
        layers=args.layers,
        apertures=args.apertures,
        n_train=args.n_train,
        n_examples=args.n_examples,
        output_dir=args.output_dir,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
