#!/usr/bin/env python3
"""
Conceptor Matrix Extraction for Emotion Steering

Collects per-sample last-token hidden states from emotion-conditioned prompts,
then computes C(R, alpha) = R(R + alpha^-2 * I)^-1 per layer per emotion.

Saves conceptor matrices as .pt files for use with conceptor_emotion_evaluation.py.

Usage:
    python conceptor_extraction.py --aperture 0.1 --num_queries 50
    python conceptor_extraction.py --aperture 0.5 --output_dir ./conceptors_a0.5

Reference: arXiv:2410.16314 - "Steering Large Language Models using Conceptors"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

import torch
import numpy as np
from tqdm import tqdm
import warnings

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Error: transformers not found. Install with: pip install transformers torch")
    sys.exit(1)

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conceptor_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_conceptor(X: torch.Tensor, aperture: float) -> torch.Tensor:
    """
    C(R, alpha) = R(R + alpha^-2 * I)^-1  via eigendecomposition of R = X.T @ X / n

    Args:
        X: (n_samples, hidden_dim) float32 on CPU
        aperture: alpha parameter (paper recommends 0.1)

    Returns:
        C: (hidden_dim, hidden_dim) float32 symmetric matrix with eigenvalues in [0, 1]
    """
    X = X.float()
    R = torch.matmul(X.T, X) / X.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(R)   # symmetric PSD -> eigh is stable
    eigenvalues = eigenvalues.clamp(min=0.0)           # numerical safety
    scale = eigenvalues / (eigenvalues + aperture ** -2)
    return (eigenvectors * scale @ eigenvectors.T).float()


class ConceptorExtractor:
    """
    Extracts conceptor matrices from emotion-conditioned Qwen activations.

    For each emotion, runs forward passes with emotion-conditioned system prompts,
    captures last-token hidden states at each transformer layer, then builds
    C(R, alpha) from the empirical covariance of those activations.
    """

    EMOTIONS = ["joy", "anger", "disgust", "fear", "sadness"]

    NEUTRAL_PROMPT = (
        "You are an AI assistant. Respond neutrally and objectively to the user's query "
        "without showing any particular emotion. Maintain a balanced and factual tone."
    )

    EMOTION_PROMPTS = {
        "joy": "Respond with strong joy emotion. Express clear joy in your tone, word choice, and perspective.",
        "anger": "Respond with strong anger emotion. Express clear anger in your tone, word choice, and perspective.",
        "disgust": "Respond with strong disgust emotion. Express clear disgust in your tone, word choice, and perspective.",
        "fear": "Respond with strong fear emotion. Express clear fear in your tone, word choice, and perspective.",
        "sadness": "Respond with strong sadness emotion. Express clear sadness in your tone, word choice, and perspective.",
    }

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str = "cuda",
                 torch_dtype: torch.dtype = torch.float32):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.hooks: List = []

        logger.info(f"Initializing ConceptorExtractor: {model_name}")
        self._load_model()
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        logger.info(f"Model: {self.num_layers} layers, hidden_size={self.hidden_size}")

    def __del__(self):
        if hasattr(self, 'hooks') and self.hooks:
            self._remove_hooks()

    def _load_model(self):
        logger.info("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if not self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded: {self.model.num_parameters():,} parameters")

    def _format_prompt(self, system_message: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        if self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _collect_raw(self,
                     system_prompt: str,
                     queries: List[str],
                     num_queries: int,
                     desc: str = "Collecting") -> Dict[int, torch.Tensor]:
        """
        Run forward passes capturing last-token hidden states at each transformer layer.

        Returns:
            Dict[layer_num -> Tensor(n_queries, hidden_dim)] float32 on CPU
        """
        storage: Dict[int, List[torch.Tensor]] = {l: [] for l in range(self.num_layers)}

        def make_hook(layer_num: int):
            def hook(module, input_tensor, output_tensor):
                h = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                storage[layer_num].append(h[0, -1, :].detach().cpu().float())
                return output_tensor
            return hook

        registered = 0
        for name, module in self.model.named_modules():
            if 'layers.' in name and name.count('.') == 2:
                try:
                    layer_num = int(name.split('layers.')[1].split('.')[0])
                except (ValueError, IndexError):
                    continue
                if 0 <= layer_num < self.num_layers:
                    self.hooks.append(module.register_forward_hook(make_hook(layer_num)))
                    registered += 1

        collection_queries = queries[:num_queries]
        try:
            with torch.no_grad():
                for query in tqdm(collection_queries, desc=desc):
                    prompt = self._format_prompt(system_prompt, query)
                    inputs = self.tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=2048
                    ).to(self.device)
                    self.model(**inputs)
        finally:
            self._remove_hooks()

        return {
            layer_num: torch.stack(samples)
            for layer_num, samples in storage.items()
            if samples
        }

    def collect_emotion_activations(self,
                                     queries: List[str],
                                     emotion: str,
                                     num_queries: int) -> Dict[int, torch.Tensor]:
        """Collect raw emotion-conditioned activations (n_queries × hidden_dim per layer)."""
        logger.info(f"Collecting activations for {emotion}...")
        return self._collect_raw(
            system_prompt=self.EMOTION_PROMPTS[emotion],
            queries=queries,
            num_queries=num_queries,
            desc=f"  Collecting {emotion}"
        )

    def build_and_save_conceptors(self,
                                   emotion_queries: Dict[str, List[str]],
                                   aperture: float,
                                   num_queries: int,
                                   output_dir: str):
        """
        Build and save conceptor matrices for all emotions.

        Saves:
            {output_dir}/{emotion}_layer_{n}.pt  for each emotion and layer
            {output_dir}/metadata.json
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Building conceptors: aperture={aperture}, queries_per_emotion={num_queries}")
        logger.info(f"Output directory: {output_dir}")

        eigenvalue_stats = {}

        for emotion in self.EMOTIONS:
            if emotion not in emotion_queries:
                logger.warning(f"No queries for {emotion}, skipping")
                continue

            # Collect raw emotion-conditioned activations
            X_raw = self.collect_emotion_activations(emotion_queries[emotion], emotion, num_queries)

            stats = []
            logger.info(f"Computing {len(X_raw)} conceptors for {emotion} (mean-centred)...")

            for layer_num, X in tqdm(sorted(X_raw.items()), desc=f"Conceptors {emotion}"):
                X_cpu = X.cpu().float()                    # (n, hidden_dim)

                # Per-emotion training mean  µ_train = (1/N) Σ h_l(d)
                mu = X_cpu.mean(dim=0)                     # (hidden_dim,)

                # Mean-centre before computing R so C captures variation, not the mean
                X_mc = X_cpu - mu                          # (n, hidden_dim)
                C = compute_conceptor(X_mc, aperture)      # (hidden_dim, hidden_dim)

                ev = torch.linalg.eigvalsh(C)
                stats.append({
                    "layer": layer_num,
                    "max_eigenvalue": float(ev.max()),
                    "mean_eigenvalue": float(ev.mean()),
                    "effective_rank": int((ev > 0.01).sum()),
                    "mu_norm": float(torch.linalg.norm(mu))
                })

                # Save C (conceptor) and µ (per-emotion training mean)
                torch.save(C,  output_path / f"{emotion}_layer_{layer_num}.pt")
                torch.save(mu, output_path / f"{emotion}_layer_{layer_num}_mu.pt")

            eigenvalue_stats[emotion] = stats

            avg_max  = np.mean([s["max_eigenvalue"]  for s in stats])
            avg_rank = np.mean([s["effective_rank"]  for s in stats])
            avg_mu   = np.mean([s["mu_norm"]         for s in stats])
            logger.info(f"{emotion}: avg_max_eigenvalue={avg_max:.4f}, avg_effective_rank={avg_rank:.1f}, avg_mu_norm={avg_mu:.4f}")

        metadata = {
            "model_name": self.model_name,
            "aperture": aperture,
            "num_collection_queries": num_queries,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "emotions": self.EMOTIONS,
            "build_timestamp": datetime.now().isoformat(),
            "method": "mean_centred_conceptor",
            "activation_type": "X_emotion mean-centred per emotion",
            "steering_formula": "h' = h + beta * C @ (h - mu_train)",
            "eigenvalue_stats": eigenvalue_stats
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Conceptors saved to {output_dir}")
        logger.info(f"Run evaluation: python conceptor_emotion_evaluation.py --conceptors_dir {output_dir}")


def load_emotion_queries(path: str) -> Dict[str, List[str]]:
    """Load EmotionQuery.json (dict keyed by emotion) for activation collection."""
    logger.info(f"Loading emotion queries from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        result = {k: v for k, v in data.items() if isinstance(v, list)}
        total = sum(len(v) for v in result.values())
        logger.info(f"Loaded {total} queries across emotions: {list(result.keys())}")
        return result
    elif isinstance(data, list):
        logger.warning("Flat list format detected; using same queries for all emotions")
        return {e: data for e in ConceptorExtractor.EMOTIONS}
    else:
        raise ValueError(f"Unexpected JSON format in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build conceptor matrices from emotion-conditioned Qwen activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--aperture", type=float, default=0.1,
                        help="Aperture α for C(R,α) — controls selectivity (0.1 per paper)")
    parser.add_argument("--num_queries", type=int, default=50,
                        help="Activation collection queries per emotion")
    parser.add_argument("--output_dir", type=str, default="./conceptors",
                        help="Directory to save .pt conceptor files")
    parser.add_argument("--emotion_queries_file", type=str, default="EmotionQuery.json",
                        help="Per-emotion queries file for collection")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    logger.info("=" * 70)
    logger.info("CONCEPTOR MATRIX EXTRACTION")
    logger.info(f"  Model:   {args.model_name}")
    logger.info(f"  Aperture: {args.aperture}")
    logger.info(f"  Queries/emotion: {args.num_queries}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 70)

    try:
        emotion_queries = load_emotion_queries(args.emotion_queries_file)

        extractor = ConceptorExtractor(model_name=args.model_name, device=args.device)

        start = time.time()
        extractor.build_and_save_conceptors(
            emotion_queries=emotion_queries,
            aperture=args.aperture,
            num_queries=args.num_queries,
            output_dir=args.output_dir
        )
        logger.info(f"Extraction complete in {time.time() - start:.1f}s")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
