#!/usr/bin/env python3
"""
Conceptor-Based Emotion Steering Evaluation

Loads pre-built conceptor matrices and evaluates emotion steering using:
    h'_l = beta_c * C_l * h_l  (projective, not additive)

where C_l = C(R, alpha) = R(R + alpha^-2 * I)^-1 and R = X^T X / n
built from emotion-conditioned activations by conceptor_extraction.py.

Compares steered vs. baseline generation using BART-large-MNLI confidence.
Output CSV columns match emotion_vector_evaluation.py for direct comparison.

Usage:
    # Step 1: build conceptors
    python conceptor_extraction.py --aperture 0.1 --num_queries 50

    # Step 2: evaluate
    python conceptor_emotion_evaluation.py --beta 20.0 --num_queries 100
    python conceptor_emotion_evaluation.py --beta 10.0 --emotion joy --num_queries 100

Reference: arXiv:2410.16314 - "Steering Large Language Models using Conceptors"
"""

import argparse
import json
import os
import sys
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

import torch
import numpy as np
from tqdm import tqdm
import warnings

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    print("Error: transformers not found. Install with: pip install transformers torch")
    sys.exit(1)

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conceptor_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConceptorEmotionEvaluator:
    """
    Evaluates conceptor-based emotion steering vs. baseline generation.

    Steering: h'_l = beta_c * C_l * h_l applied at last token position, all layers.
    C_l loaded from .pt files produced by conceptor_extraction.py.
    """

    EMOTIONS = ["joy", "anger", "disgust", "fear", "sadness"]

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 classifier_name: str = "facebook/bart-large-mnli",
                 device: str = "cuda",
                 torch_dtype: torch.dtype = torch.bfloat16):
        self.model_name = model_name
        self.classifier_name = classifier_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.hooks: List = []
        # current_emotion_data[layer_num] = (C, mu) for the active emotion
        self.current_emotion_data: Optional[Dict[int, tuple]] = None
        self.current_beta: float = 0.0

        logger.info(f"Initializing ConceptorEmotionEvaluator")
        logger.info(f"Generation model: {model_name}")
        logger.info(f"Classifier: {classifier_name}")

        self._load_generation_model()
        self._load_classification_model()
        logger.info("Evaluator ready")

    def __del__(self):
        if hasattr(self, 'hooks') and self.hooks:
            self._remove_hooks()

    def _load_generation_model(self):
        logger.info("Loading generation model...")
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
            output_hidden_states=True,
            return_dict_in_generate=True,
            low_cpu_mem_usage=True,
        )
        if not self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Generation model loaded: {self.model.num_parameters():,} parameters")

    def _load_classification_model(self):
        logger.info("Loading BART-large-MNLI classifier...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.classifier_name,
            device=0 if self.device.startswith("cuda") and torch.cuda.is_available() else -1
        )
        logger.info("Classifier loaded")

    def _format_qwen_prompt(self, system_message: str, user_message: str) -> str:
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

    def load_conceptors(self, conceptors_dir: str) -> Dict[str, Dict[int, tuple]]:
        """
        Load conceptor matrices C and per-emotion training means µ from .pt files.

        Returns:
            emotion_data[emotion][layer_num] = (C: Tensor(d,d), mu: Tensor(d,)) on CPU
        """
        conceptors_path = Path(conceptors_dir)
        if not conceptors_path.exists():
            raise FileNotFoundError(
                f"Conceptors directory not found: {conceptors_dir}\n"
                f"Run: python conceptor_extraction.py --output_dir {conceptors_dir}"
            )

        metadata_file = conceptors_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)
            logger.info(f"Conceptors: aperture={meta.get('aperture')}, "
                        f"method={meta.get('method', 'unknown')}, "
                        f"queries={meta.get('num_collection_queries')}, "
                        f"built={meta.get('build_timestamp', 'unknown')[:19]}")

        emotion_data = {}
        for emotion in self.EMOTIONS:
            layers = {}
            layer_num = 0
            while True:
                c_file  = conceptors_path / f"{emotion}_layer_{layer_num}.pt"
                mu_file = conceptors_path / f"{emotion}_layer_{layer_num}_mu.pt"
                if not c_file.exists():
                    break
                C  = torch.load(c_file,  map_location='cpu')
                mu = torch.load(mu_file, map_location='cpu') if mu_file.exists() else torch.zeros(C.shape[0])
                layers[layer_num] = (C, mu)
                layer_num += 1

            if layers:
                emotion_data[emotion] = layers
                logger.info(f"Loaded {len(layers)} (C, mu) pairs for {emotion}")
            else:
                logger.warning(f"No .pt files for {emotion} in {conceptors_dir} — re-run extraction")

        if not emotion_data:
            raise FileNotFoundError(
                f"No conceptor .pt files in {conceptors_dir}. "
                "Re-run: python conceptor_extraction.py"
            )
        return emotion_data

    def _create_steering_hook(self, layer_num: int):
        """
        h'_l = h_l + beta * C_l * (h_l - mu_train_l)

        Mean-centres h w.r.t. the per-emotion training mean before projecting
        through the conceptor, then adds the result back (additive, content-preserving).
        """
        def hook(module, input_tensor, output_tensor):
            if self.current_emotion_data is None or layer_num not in self.current_emotion_data:
                return output_tensor

            h = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            C, mu = self.current_emotion_data[layer_num]
            C  = C.to(device=h.device,  dtype=h.dtype)
            mu = mu.to(device=h.device, dtype=h.dtype)

            h = h.clone()
            h_mc = h[:, -1, :] - mu          # mean-centre w.r.t. training distribution
            h[:, -1, :] += self.current_beta * (h_mc @ C)   # additive projection

            if isinstance(output_tensor, tuple):
                return (h,) + output_tensor[1:]
            return h
        return hook

    def _register_steering_hooks(self):
        """Register steering hooks on transformer layers (model.layers.{n} only)."""
        registered = 0
        for name, module in self.model.named_modules():
            if 'layers.' in name and name.count('.') == 2:
                try:
                    layer_num = int(name.split('layers.')[1].split('.')[0])
                except (ValueError, IndexError):
                    continue
                self.hooks.append(
                    module.register_forward_hook(self._create_steering_hook(layer_num))
                )
                registered += 1
        logger.debug(f"Registered {registered} steering hooks")

    def generate_with_conceptor(self,
                                 prompt: str,
                                 emotion_data: Optional[Dict[int, tuple]] = None,
                                 beta: float = 0.0,
                                 max_new_tokens: int = 50) -> str:
        """Generate with optional conceptor steering. emotion_data=None → baseline."""
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=2048, padding=False
            ).to(self.device)

            if emotion_data and beta != 0.0:
                self.current_emotion_data = emotion_data
                self.current_beta = beta
                self._register_steering_hooks()
            else:
                self.current_emotion_data = None
                self.current_beta = 0.0

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            if emotion_data and beta != 0.0:
                self._remove_hooks()

            return self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

        except Exception as e:
            logger.error(f"Generation error: {e}")
            if self.hooks:
                self._remove_hooks()
            return ""

    def generate_batch(self,
                       prompts: List[str],
                       emotion_data: Optional[Dict[int, tuple]] = None,
                       beta: float = 0.0,
                       max_new_tokens: int = 50,
                       batch_size: int = 8,
                       greedy: bool = False) -> List[str]:
        """
        Generate for all prompts in batches with optional conceptor steering.
        Left-padded batching keeps generated-token offsets consistent across the batch.
        """
        results = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", leave=False):
            batch = prompts[i:i + batch_size]
            try:
                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048
                ).to(self.device)
                input_len = inputs['input_ids'].shape[1]

                if emotion_data and beta != 0.0:
                    self.current_emotion_data = emotion_data
                    self.current_beta = beta
                    self._register_steering_hooks()
                else:
                    self.current_emotion_data = None
                    self.current_beta = 0.0

                gen_kwargs: Dict[str, Any] = dict(
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                if greedy:
                    gen_kwargs["do_sample"] = False
                else:
                    gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                logger.error(f"Batch generation error (batch {i // batch_size}): {e}")
                results.extend([""] * len(batch))
                continue
            finally:
                if self.hooks:
                    self._remove_hooks()

            for seq in outputs:
                results.append(
                    self.tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
                )
        return results

    def classify_emotion_confidence_batch(self,
                                           texts: List[str],
                                           classifier_batch_size: int = 32) -> List[Dict[str, float]]:
        """Classify a list of texts in one batched BART-MNLI call.
        Empty/whitespace texts are skipped and returned as {} to avoid BART crashes."""
        candidate_labels = ["joy", "anger", "disgust", "fear", "sadness", "neutral"]
        output: List[Dict[str, float]] = [{} for _ in texts]

        # Only classify non-empty texts
        valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid:
            logger.warning("All texts are empty, skipping classification")
            return output

        indices, valid_texts = zip(*valid)
        try:
            results = self.classifier(list(valid_texts), candidate_labels, batch_size=classifier_batch_size)
            for i, r in zip(indices, results):
                output[i] = {label: float(score) for label, score in zip(r['labels'], r['scores'])}
        except Exception as e:
            logger.error(f"Batch classification error: {e}")
        return output

    def classify_emotion_confidence(self, text: str, target_emotion: str) -> Dict[str, float]:
        """Classify emotion in text using BART-large-MNLI zero-shot classification."""
        try:
            candidate_labels = ["joy", "anger", "disgust", "fear", "sadness", "neutral"]
            result = self.classifier(text, candidate_labels)
            scores = {
                label: float(score)
                for label, score in zip(result['labels'], result['scores'])
            }
            all_scores = ", ".join(
                f"{l}: {scores.get(l, 0.0):.3f}"
                for l in ["joy", "anger", "disgust", "fear", "sadness", "neutral"]
            )
            logger.debug(f"'{text[:30]}...' -> [{all_scores}] target={target_emotion}: {scores.get(target_emotion, 0.0):.3f}")
            return scores
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {}

    def evaluate_conceptors(self,
                             emotion_data: Dict[str, Dict[int, tuple]],
                             eval_queries: List[str],
                             beta: float,
                             num_queries_per_emotion: int = 100,
                             target_emotions: Optional[List[str]] = None,
                             max_new_tokens: int = 50,
                             gen_batch_size: int = 8,
                             classifier_batch_size: int = 32,
                             greedy: bool = False) -> List[Dict[str, Any]]:
        """
        Evaluate conceptor steering vs. baseline for each emotion.

        Optimised pipeline:
          1. Generate all baseline responses once (reused across all emotions).
          2. Per emotion: pre-move (C, mu) pairs to GPU, generate enhanced responses
             in batches, then classify all 2*N texts in one BART-MNLI call.
          3. Free GPU data between emotions.

        Steering formula: h'_l = h_l + beta * C_l * (h_l - mu_train_l)

        Returns:
            List of result dicts with same keys as emotion_vector_evaluation.py.
        """
        emotions_to_test = target_emotions if target_emotions else self.EMOTIONS
        queries = eval_queries[:num_queries_per_emotion]

        logger.info(f"Evaluation: {len(emotions_to_test)} emotions x {len(queries)} queries, beta={beta}")
        logger.info(f"gen_batch_size={gen_batch_size}, classifier_batch_size={classifier_batch_size}, "
                    f"max_new_tokens={max_new_tokens}, greedy={greedy}")

        neutral_system = (
            "You are an AI assistant. Respond neutrally and objectively to the user's query "
            "without showing any particular emotion. Maintain a balanced and factual tone."
        )
        prompts = [self._format_qwen_prompt(neutral_system, q) for q in queries]

        # Step 1: generate baselines once — same neutral prompts reused for all emotions
        logger.info(f"Generating {len(prompts)} baseline responses (shared across all emotions)...")
        baseline_responses = self.generate_batch(
            prompts, emotion_data=None, beta=0.0,
            max_new_tokens=max_new_tokens, batch_size=gen_batch_size, greedy=greedy
        )

        all_results = []
        emotions_evaluated = []

        for emotion in emotions_to_test:
            if emotion not in emotion_data:
                logger.warning(f"No conceptors for {emotion}, skipping")
                continue

            emotions_evaluated.append(emotion)

            # Pre-move (C, mu) pairs to GPU/model dtype once per emotion
            gpu_data = {
                l: (C.to(device=self.device, dtype=self.torch_dtype),
                    mu.to(device=self.device, dtype=self.torch_dtype))
                for l, (C, mu) in emotion_data[emotion].items()
            }
            logger.info(f"[{emotion}] Generating {len(prompts)} enhanced responses "
                        f"({len(gpu_data)} layers on GPU)...")

            # Step 2: generate steered responses
            enhanced_responses = self.generate_batch(
                prompts, emotion_data=gpu_data, beta=beta,
                max_new_tokens=max_new_tokens, batch_size=gen_batch_size, greedy=greedy
            )

            # Step 3: batch-classify all 2*N texts in one BART-MNLI call
            all_texts = baseline_responses + enhanced_responses
            logger.info(f"[{emotion}] Classifying {len(all_texts)} responses "
                        f"(classifier batch_size={classifier_batch_size})...")
            all_conf = self.classify_emotion_confidence_batch(all_texts, classifier_batch_size)

            baseline_confs = all_conf[:len(queries)]
            enhanced_confs = all_conf[len(queries):]

            # Free GPU memory before next emotion
            del gpu_data
            torch.cuda.empty_cache()

            ts = datetime.now().isoformat()
            for query_idx, (query, baseline, enhanced, b_conf, e_conf) in enumerate(
                zip(queries, baseline_responses, enhanced_responses, baseline_confs, enhanced_confs)
            ):
                b_score = b_conf.get(emotion, 0.0)
                e_score = e_conf.get(emotion, 0.0)
                all_results.append({
                    "emotion": emotion,
                    "layer": "ALL",
                    "query_idx": query_idx,
                    "alpha": beta,
                    "baseline_max_confidence": b_score,
                    "enhanced_max_confidence": e_score,
                    "confidence_improvement": e_score - b_score,
                    "query": query,
                    "baseline_response": baseline,
                    "enhanced_response": enhanced,
                    "timestamp": ts
                })

        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY:")
        for emotion in emotions_evaluated:
            r = [x for x in all_results if x["emotion"] == emotion]
            if r:
                avg = np.mean([x["confidence_improvement"] for x in r])
                sr = float(np.mean([x["confidence_improvement"] > 0 for x in r]))
                logger.info(f"  {emotion.upper()}: {len(r)} queries, avg_improvement={avg:+.4f}, success={sr:.1%}")
        logger.info(f"Total: {len(all_results)} evaluations")
        logger.info("=" * 50)

        return all_results

    def save_evaluation_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save evaluation results to CSV (same columns as emotion_vector_evaluation.py)."""
        if not results:
            logger.warning("No results to save")
            return

        fieldnames = [
            "emotion", "layer", "query_idx", "alpha",
            "baseline_max_confidence", "enhanced_max_confidence", "confidence_improvement",
            "query", "baseline_response", "enhanced_response", "timestamp"
        ]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in fieldnames})

        size_kb = os.path.getsize(output_file) / 1024
        logger.info(f"Results saved to {output_file} ({size_kb:.2f} KB)")

    def save_emotion_summary(self,
                              results: List[Dict[str, Any]],
                              output_file: str,
                              aperture: float,
                              beta: float):
        """Save per-emotion summary statistics to JSON."""
        if not results:
            logger.warning("No results to summarize")
            return

        emotion_stats = {}
        for emotion in self.EMOTIONS:
            r = [x for x in results if x["emotion"] == emotion]
            if not r:
                continue

            improvements = [x["confidence_improvement"] for x in r]
            baselines = [x["baseline_max_confidence"] for x in r]
            enhanced_scores = [x["enhanced_max_confidence"] for x in r]
            positive = [i for i in improvements if i > 0]

            emotion_stats[emotion] = {
                "total_queries": len(r),
                "average_improvement": float(np.mean(improvements)),
                "median_improvement": float(np.median(improvements)),
                "std_improvement": float(np.std(improvements)),
                "min_improvement": float(np.min(improvements)),
                "max_improvement": float(np.max(improvements)),
                "positive_improvements": len(positive),
                "negative_improvements": len([i for i in improvements if i < 0]),
                "success_rate": float(len(positive) / len(improvements)),
                "average_baseline_confidence": float(np.mean(baselines)),
                "average_enhanced_confidence": float(np.mean(enhanced_scores)),
                "confidence_boost": float(np.mean(enhanced_scores) - np.mean(baselines)),
                "evaluation_approach": "CONCEPTOR_ALL_LAYERS"
            }

        all_imp = [x["confidence_improvement"] for x in results]
        overall = {
            "total_evaluations": len(results),
            "emotions_tested": list(emotion_stats.keys()),
            "overall_avg_improvement": float(np.mean(all_imp)),
            "overall_success_rate": float(np.mean([i > 0 for i in all_imp])),
            "best_performing_emotion": max(emotion_stats, key=lambda e: emotion_stats[e]["average_improvement"]) if emotion_stats else None,
            "worst_performing_emotion": min(emotion_stats, key=lambda e: emotion_stats[e]["average_improvement"]) if emotion_stats else None,
            "beta": beta,
            "aperture": aperture,
            "evaluation_timestamp": datetime.now().isoformat()
        }

        summary = {
            "overall_summary": overall,
            "emotion_statistics": emotion_stats,
            "evaluation_metadata": {
                "model_name": self.model_name,
                "classifier_name": self.classifier_name,
                "device": self.device,
                "aperture": aperture,
                "beta": beta,
                "method": "conceptor_projection",
                "paper_reference": "arXiv:2410.16314",
                "steering_formula": "h'_l = beta_c * C_l * h_l",
                "conceptor_formula": "C(R,alpha) = R(R + alpha^-2 * I)^-1, R = X.T @ X / n"
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        size_kb = os.path.getsize(output_file) / 1024
        logger.info(f"Summary saved to {output_file} ({size_kb:.2f} KB)")
        logger.info("=== CONCEPTOR STEERING RESULTS ===")
        for emotion, stats in emotion_stats.items():
            logger.info(f"  {emotion.upper()}: success={stats['success_rate']:.1%}, avg_improvement={stats['average_improvement']:+.4f}")
        logger.info(f"OVERALL: {overall['overall_success_rate']:.1%} success, avg={overall['overall_avg_improvement']:+.4f}")
        logger.info(f"BEST: {overall['best_performing_emotion']}")


def load_eval_queries(path: str) -> List[str]:
    """Load evaluation queries from EmotionQuery+.json (flat list)."""
    logger.info(f"Loading eval queries from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        logger.info(f"Loaded {len(data)} eval queries")
        return data
    elif isinstance(data, dict):
        queries = []
        for v in data.values():
            if isinstance(v, list):
                queries.extend(v)
        logger.info(f"Flattened {len(queries)} eval queries from dict format")
        return queries
    raise ValueError(f"Unexpected format in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate conceptor-based emotion steering (arXiv:2410.16314)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--beta", type=float, default=20.0,
                        help="β_c scaling in h'=β·C·h (steering intensity, tune if outputs degenerate)")
    parser.add_argument("--aperture", type=float, default=0.1,
                        help="Aperture α used during extraction (logged in summary metadata)")
    parser.add_argument("--conceptors_dir", type=str, default="./conceptors",
                        help="Directory with .pt conceptor files from conceptor_extraction.py")
    parser.add_argument("--eval_queries_file", type=str, default="EmotionQuery+.json",
                        help="Flat-list queries file for evaluation")
    parser.add_argument("--num_queries", type=int, default=100,
                        help="Evaluation queries per emotion")
    parser.add_argument("--emotion", type=str, default="all",
                        choices=["joy", "anger", "disgust", "fear", "sadness", "all"])
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output CSV file (auto-generated if not specified)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gen_batch_size", type=int, default=8,
                        help="Qwen generation batch size (RTX 3080 10GB: 8 safe, try 16)")
    parser.add_argument("--classifier_batch_size", type=int, default=32,
                        help="BART-MNLI classification batch size")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Max tokens per response (30 is ~40%% faster than 50)")
    parser.add_argument("--greedy", action="store_true",
                        help="Greedy decoding instead of sampling (faster, deterministic)")

    args = parser.parse_args()

    if args.output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        emotion_suffix = f"_{args.emotion}" if args.emotion != "all" else ""
        args.output_file = f"conceptor_evaluation_beta{args.beta}_aperture{args.aperture}{emotion_suffix}_{ts}.csv"

    summary_file = args.output_file.replace('.csv', '_summary.json')

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger.info("=" * 70)
    logger.info("CONCEPTOR-BASED EMOTION STEERING EVALUATION")
    logger.info("Reference: arXiv:2410.16314")
    logger.info(f"  Model:    {args.model_name}")
    logger.info(f"  Beta (β): {args.beta}")
    logger.info(f"  Aperture: {args.aperture} (from extraction)")
    logger.info(f"  Queries/emotion: {args.num_queries}")
    logger.info(f"  Conceptors: {args.conceptors_dir}")
    logger.info(f"  Gen batch_size: {args.gen_batch_size} | Classifier batch_size: {args.classifier_batch_size}")
    logger.info(f"  Max new tokens: {args.max_new_tokens}")
    logger.info(f"  Output: {args.output_file}")
    logger.info("=" * 70)

    try:
        eval_queries = load_eval_queries(args.eval_queries_file)

        evaluator = ConceptorEmotionEvaluator(
            model_name=args.model_name,
            device=args.device
        )

        emotion_data = evaluator.load_conceptors(args.conceptors_dir)

        target_emotions = None if args.emotion == "all" else [args.emotion]

        start = time.time()
        results = evaluator.evaluate_conceptors(
            emotion_data=emotion_data,
            eval_queries=eval_queries,
            beta=args.beta,
            num_queries_per_emotion=args.num_queries,
            target_emotions=target_emotions,
            max_new_tokens=args.max_new_tokens,
            gen_batch_size=args.gen_batch_size,
            classifier_batch_size=args.classifier_batch_size,
            greedy=args.greedy,
        )
        elapsed = time.time() - start
        logger.info(f"Evaluation completed in {elapsed:.1f}s")

        evaluator.save_evaluation_results(results, args.output_file)
        evaluator.save_emotion_summary(results, summary_file, args.aperture, args.beta)

        if results:
            avg_imp = float(np.mean([r["confidence_improvement"] for r in results]))
            pos = sum(1 for r in results if r["confidence_improvement"] > 0)
            logger.info("=" * 70)
            logger.info("COMPLETE")
            logger.info(f"  Total evaluations: {len(results)}")
            logger.info(f"  Average improvement: {avg_imp:+.4f}")
            logger.info(f"  Positive: {pos}/{len(results)} ({pos/len(results)*100:.1f}%)")
            logger.info(f"  Results: {os.path.abspath(args.output_file)}")
            logger.info(f"  Summary: {os.path.abspath(summary_file)}")
            logger.info("=" * 70)

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
