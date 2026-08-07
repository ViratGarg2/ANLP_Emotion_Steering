#!/usr/bin/env python3
"""
Emotion Vector Extraction for Qwen-2.5-1.5B-Instruct Model

This script implements the emotion vector extraction methodology from:
"From Rational Answers to Emotional Resonance: The Role of Controllable Emotion Generation in Language Models"
(arXiv:2502.04075v2)

The script extracts hidden state representations for different emotions by computing
the difference between emotion-conditioned and neutral responses, following the
paper's methodology.

Two capture modes (--mode):
    averaged    (default) Save only the query-averaged vector per (emotion,
                layer), as JSON. Original behaviour, byte-compatible output.
    per_prompt  Save the per-query hidden states as compressed .npz — the
                emotional and neutral states and their difference, one row per
                query. This is what makes a real per-layer t-SNE possible, since
                the averaged mode leaves only one point per (emotion, layer).
    both        Save both.

Usage:
    python qwen_emotion_extraction.py --num_queries 50 --output_dir ./emotion_vectors
    python qwen_emotion_extraction.py --num_queries 100 --batch_size 4 --device cuda:0
    python qwen_emotion_extraction.py --num_queries 100 --mode both
    python qwen_emotion_extraction.py --num_queries 100 --mode per_prompt --per_prompt_dtype float32

Author: Based on arXiv:2502.04075v2 methodology
Date: November 2025
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

import torch
import numpy as np
from tqdm import tqdm
import warnings

# Transformers imports
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        GenerationConfig
    )
except ImportError:
    print("Error: transformers library not found. Please install with: pip install transformers torch")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QwenEmotionVectorExtractor:
    """
    Emotion Vector Extractor for Qwen-2.5-1.5B-Instruct model
    
    Implements the methodology from arXiv:2502.04075v2:
    1. Generate responses under emotional and neutral conditions
    2. Extract hidden states from all layers during generation
    3. Compute emotion vectors as difference between emotional and neutral states
    4. Average across queries for each emotion and layer
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str = "cuda",
                 max_length: int = 2048,
                 torch_dtype: torch.dtype = torch.float32):
        """
        Initialize the Qwen Emotion Vector Extractor
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on ('cuda', 'cpu', or specific GPU)
            max_length: Maximum sequence length for tokenization
            torch_dtype: PyTorch data type for model weights
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.torch_dtype = torch_dtype

        # Per-prompt capture. When enabled, extract_emotion_vectors() keeps the
        # per-query hidden states instead of discarding them after averaging.
        # Set by main() from --mode; buffers are flushed per emotion.
        self.collect_per_prompt = False
        self.per_prompt_dtype = np.float16
        self.per_prompt_dir = None
        self.per_prompt_counts = {}

        logger.info(f"Initializing Qwen Emotion Vector Extractor")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Max Length: {max_length}")
        logger.info(f"Data Type: {torch_dtype}")
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Define emotion categories from the paper
        self.emotions = ["joy", "anger", "disgust", "fear", "sadness"]
        
        logger.info(f"Extractor initialized successfully")
        logger.info(f"Target emotions: {self.emotions}")
    
    def _load_model_and_tokenizer(self):
        """Load Qwen model and tokenizer with specified configurations"""
        
        logger.info("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Configure padding token for Qwen
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info(f"Tokenizer loaded - Vocab size: {len(self.tokenizer)}")
            logger.info(f"Chat template available: {self.tokenizer.chat_template is not None}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        logger.info("Loading model...")
        try:
            # output_hidden_states / return_dict_in_generate are GENERATION
            # kwargs, not constructor kwargs. transformers >= 4.5x forwards
            # unrecognised kwargs into the model __init__, which raises
            # TypeError. Both are already passed explicitly at each
            # generate()/forward() call site, so they belong there, not here.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device.startswith("cuda") else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Move to device if not using device_map
            if not self.device.startswith("cuda") or self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Model loaded successfully!")
            logger.info(f"Parameters: {self.model.num_parameters():,}")
            logger.info(f"Layers: {self.model.config.num_hidden_layers}")
            logger.info(f"Hidden size: {self.model.config.hidden_size}")
            logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
            
            if torch.cuda.is_available():
                logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _format_qwen_prompt(self, system_message: str, user_message: str) -> str:
        """
        Format prompt using Qwen's chat template for proper conversation structure
        
        Args:
            system_message: System-level instruction
            user_message: User query/message
            
        Returns:
            Formatted prompt string
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Use Qwen's chat template if available
        if self.tokenizer.chat_template:
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return formatted_prompt
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback")
        
        # Fallback formatting
        return f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
    
    def _extract_hidden_states_from_generation(self, 
                                             prompt: str, 
                                             max_new_tokens: int = 50) -> Optional[List[torch.Tensor]]:
        """
        Extract hidden states during text generation with mean-pooling across all tokens
        
        This method extracts hidden representations from each layer during generation,
        then applies mean-pooling across all tokens to create a single vector per layer.
        
        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            List of mean-pooled hidden states for each layer, or None if failed
        """
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=self.max_length,
                padding=False
            ).to(self.device)
            
            # Generate response with hidden states
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    use_cache=False,  # Important for getting hidden states
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract hidden states from the generation
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                all_layer_states = []
                
                # Process each generation step
                for step_hidden_states in outputs.hidden_states:
                    step_layer_averages = []
                    
                    # Process each layer in this step
                    for layer_idx, layer_hidden_state in enumerate(step_hidden_states):
                        # Mean-pool across all tokens in the sequence (dim=1 is sequence length)
                        # Shape: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
                        mean_pooled_state = layer_hidden_state.mean(dim=1, keepdim=False).squeeze(0)
                        step_layer_averages.append(mean_pooled_state.cpu().float())
                    
                    all_layer_states.append(step_layer_averages)
                
                # Average across generation steps for each layer (mean-pool across time steps)
                if all_layer_states:
                    final_layer_states = []
                    num_layers = len(all_layer_states[0])
                    
                    for layer_idx in range(num_layers):
                        layer_states_across_steps = [step_states[layer_idx] for step_states in all_layer_states]
                        if layer_states_across_steps:
                            # Mean-pool across all generation steps
                            avg_layer_state = torch.stack(layer_states_across_steps).mean(dim=0)
                            final_layer_states.append(avg_layer_state)
                    
                    return final_layer_states
            
            # Fallback: use hidden states from a single forward pass
            return self._extract_hidden_states_forward_pass(prompt)
            
        except Exception as e:
            logger.error(f"Error in hidden state extraction: {e}")
            return None
    
    def _extract_hidden_states_forward_pass(self, prompt: str) -> Optional[List[torch.Tensor]]:
        """Fallback method using a single forward pass with mean-pooling across all tokens"""
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Extract and mean-pool hidden states across all tokens
            hidden_states = outputs.hidden_states  # Tuple of layer outputs
            mean_pooled_states = []
            
            for layer_hidden_state in hidden_states:
                # Mean-pool across all tokens in the sequence (dim=1 is sequence length)
                # Shape: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
                mean_pooled_state = layer_hidden_state.mean(dim=1, keepdim=False).squeeze(0).cpu().float()
                mean_pooled_states.append(mean_pooled_state)
            
            return mean_pooled_states
            
        except Exception as e:
            logger.error(f"Error in forward pass extraction: {e}")
            return None
    
    def extract_emotion_vectors(self, 
                              emotion_queries: Dict[str, List[str]], 
                              num_queries_per_emotion: int) -> Tuple[Dict[str, List[torch.Tensor]], 
                                                                    Optional[List[torch.Tensor]]]:
        """
        Extract emotion vectors for each emotion using the paper's methodology
        
        Args:
            emotion_queries: Dictionary mapping emotions to lists of queries
            num_queries_per_emotion: Number of queries to process per emotion
            
        Returns:
            Tuple of (emotion_vectors_dict, base_vector_list)
        """
        logger.info("Starting emotion vector extraction...")
        logger.info(f"Processing {len(self.emotions)} emotions × {num_queries_per_emotion} queries each")
        
        emotion_vectors = {}
        all_difference_vectors = []
        
        for emotion in self.emotions:
            if emotion not in emotion_queries:
                logger.warning(f"Emotion '{emotion}' not found in emotion_queries, skipping")
                continue
                
            logger.info(f"Processing emotion: {emotion.upper()}")
            
            queries = emotion_queries[emotion][:num_queries_per_emotion]
            emotion_difference_vectors = []

            # Per-prompt buffer for this emotion only — flushed to disk and
            # freed at the end of the emotion so peak RAM stays at one
            # emotion's worth rather than all five.
            per_prompt_buffer = [] if self.collect_per_prompt else None

            for query_idx, query in enumerate(tqdm(queries, desc=f"Extracting {emotion}")):
                try:
                    # Create system messages for emotional and neutral responses
                    # Following the paper's methodology for emotion conditioning
                    emotional_system = (
                        f"You are an AI assistant. Respond with strong {emotion} emotion to the user's query. "
                        f"Express clear {emotion} in your tone, word choice, and perspective."
                    )
                    neutral_system = (
                        "You are an AI assistant. Respond neutrally and objectively to the user's query "
                        "without showing any particular emotion. Maintain a balanced and factual tone."
                    )
                    
                    # Format prompts using Qwen chat template
                    emotional_prompt = self._format_qwen_prompt(emotional_system, query)
                    neutral_prompt = self._format_qwen_prompt(neutral_system, query)
                    
                    # Extract hidden states for emotional response
                    emotional_states = self._extract_hidden_states_from_generation(emotional_prompt)
                    
                    # Extract hidden states for neutral response
                    neutral_states = self._extract_hidden_states_from_generation(neutral_prompt)
                    
                    # Compute difference vectors (emotional - neutral) for each layer
                    if (emotional_states and neutral_states and 
                        len(emotional_states) == len(neutral_states)):
                        
                        difference_vectors = []
                        for emo_state, neu_state in zip(emotional_states, neutral_states):
                            diff_vector = emo_state - neu_state
                            difference_vectors.append(diff_vector)
                        
                        emotion_difference_vectors.append(difference_vectors)

                        # Keep the per-query states. Nothing extra is computed
                        # here — these are the same tensors the average is
                        # built from, they just stop being thrown away.
                        if per_prompt_buffer is not None:
                            per_prompt_buffer.append({
                                "query_idx": query_idx,
                                "query": query,
                                "emotional": torch.stack(emotional_states).numpy(),
                                "neutral": torch.stack(neutral_states).numpy(),
                                "diff": torch.stack(difference_vectors).numpy(),
                            })

                        logger.debug(f"Query {query_idx+1}/{len(queries)}: Success")

                    else:
                        logger.warning(f"Query {query_idx+1}/{len(queries)}: Failed - dimension mismatch")
                        continue
                
                except Exception as e:
                    logger.error(f"Query {query_idx+1}/{len(queries)}: Error - {str(e)}")
                    continue
            
            # Average difference vectors across queries for this emotion
            if emotion_difference_vectors:
                num_layers = len(emotion_difference_vectors[0])
                averaged_emotion_vectors = []
                
                for layer_idx in range(num_layers):
                    layer_vectors = [query_diffs[layer_idx] for query_diffs in emotion_difference_vectors]
                    if layer_vectors:
                        avg_layer_vector = torch.stack(layer_vectors).mean(dim=0)
                        averaged_emotion_vectors.append(avg_layer_vector)
                
                emotion_vectors[emotion] = averaged_emotion_vectors
                all_difference_vectors.extend(emotion_difference_vectors)
                
                success_rate = len(emotion_difference_vectors) / len(queries) * 100
                logger.info(f"{emotion}: {len(emotion_difference_vectors)}/{len(queries)} queries successful ({success_rate:.1f}%)")
            else:
                logger.error(f"{emotion}: No successful extractions")

            # Flush this emotion's per-prompt arrays and release the buffer
            if per_prompt_buffer:
                self._flush_per_prompt(emotion, per_prompt_buffer)
                per_prompt_buffer.clear()

        # Compute base emotion vector (average across all emotions and queries)
        base_vector = None
        if all_difference_vectors:
            logger.info("Computing base emotion vector...")
            num_layers = len(all_difference_vectors[0])
            base_layer_vectors = []
            
            for layer_idx in range(num_layers):
                all_layer_vectors = []
                for query_diffs in all_difference_vectors:
                    if layer_idx < len(query_diffs):
                        all_layer_vectors.append(query_diffs[layer_idx])
                
                if all_layer_vectors:
                    avg_base_vector = torch.stack(all_layer_vectors).mean(dim=0)
                    base_layer_vectors.append(avg_base_vector)
            
            base_vector = base_layer_vectors
            logger.info(f"Base vector computed with {len(base_layer_vectors)} layers")
        
        return emotion_vectors, base_vector
    
    def _flush_per_prompt(self, emotion: str, buffer: List[Dict]):
        """
        Write one emotion's per-query hidden states to a compressed .npz.

        Called once per emotion from extract_emotion_vectors() so the buffer can
        be released immediately afterwards. JSON is not used here: the arrays
        run to hundreds of MB and JSON would balloon that several-fold.

        Arrays are (n_queries, n_layers, hidden_dim). Rows are aligned with the
        stored `queries` / `query_idx` arrays rather than assumed positional,
        because failed queries are skipped by the caller's except/continue.

        Args:
            emotion: Emotion name, used in the output filename
            buffer: List of per-query dicts from extract_emotion_vectors()
        """
        if not buffer:
            logger.warning(f"{emotion}: nothing to write for per-prompt mode")
            return

        out_dir = Path(self.per_prompt_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / f"Qwen25_1B5_{emotion}_per_prompt.npz"

        dt = self.per_prompt_dtype
        arrays = {
            "diff": np.stack([r["diff"] for r in buffer]).astype(dt),
            "emotional": np.stack([r["emotional"] for r in buffer]).astype(dt),
            "neutral": np.stack([r["neutral"] for r in buffer]).astype(dt),
            "queries": np.array([r["query"] for r in buffer], dtype=object),
            "query_idx": np.array([r["query_idx"] for r in buffer], dtype=np.int32),
        }

        np.savez_compressed(output_file, **arrays)
        self.per_prompt_counts[emotion] = len(buffer)

        size_mb = os.path.getsize(output_file) / 1e6
        logger.info(
            f"Saved {emotion} per-prompt states: {arrays['diff'].shape} "
            f"({np.dtype(dt).name}, {size_mb:.1f} MB)"
        )

    def save_per_prompt_manifest(self, output_dir: str, num_queries_per_emotion: int,
                                 emotion_queries_file: str):
        """
        Write the manifest describing the per-prompt .npz set.

        Args:
            output_dir: Directory holding the .npz files
            num_queries_per_emotion: Number of queries requested per emotion
            emotion_queries_file: Source query file, recorded for provenance
        """
        if not self.per_prompt_counts:
            logger.warning("No per-prompt data was written, skipping manifest")
            return

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "model_name": self.model_name,
            "dtype": np.dtype(self.per_prompt_dtype).name,
            "n_layers": self.model.config.num_hidden_layers + 1,
            "hidden_size": self.model.config.hidden_size,
            "emotions": list(self.per_prompt_counts.keys()),
            "queries_per_emotion": self.per_prompt_counts,
            "num_queries_requested": num_queries_per_emotion,
            "emotion_queries_file": emotion_queries_file,
            "device": self.device,
            "extraction_date": datetime.now().isoformat(),
            "arrays": {
                "diff": "emotional - neutral, (n_queries, n_layers, hidden_dim)",
                "emotional": "emotion-conditioned states, same shape",
                "neutral": "neutral-conditioned states, same shape",
                "queries": "query text, row-aligned with the arrays",
                "query_idx": "index into the source query list",
            },
            "note": (
                "n_layers is num_hidden_layers + 1: index 0 is the embedding "
                "output and index k is the output of decoder block k-1. Layer "
                "keys are left as raw hidden_states indices."
            ),
        }

        manifest_file = out_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        total = sum(self.per_prompt_counts.values())
        logger.info(f"Saved per-prompt manifest: {total} queries across "
                    f"{len(self.per_prompt_counts)} emotions")

    def save_emotion_vectors(self,
                           emotion_vectors: Dict[str, List[torch.Tensor]],
                           base_vector: Optional[List[torch.Tensor]],
                           output_dir: str,
                           num_queries_per_emotion: int):
        """
        Save emotion vectors to JSON files
        
        Args:
            emotion_vectors: Dictionary of emotion vectors by emotion
            base_vector: Base emotion vector (average of all)
            output_dir: Directory to save JSON files
            num_queries_per_emotion: Number of queries used per emotion
        """
        
        def tensor_to_list(tensor):
            """Convert tensor to JSON-serializable list"""
            if isinstance(tensor, torch.Tensor):
                return tensor.cpu().numpy().tolist()
            return tensor
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving emotion vectors to: {output_dir}")
        
        # Save individual emotion vectors
        for emotion in self.emotions:
            if emotion in emotion_vectors and emotion_vectors[emotion]:
                emotion_data = {}
                
                # Save each layer's vector
                for layer_idx, vector in enumerate(emotion_vectors[emotion]):
                    emotion_data[f"layer_{layer_idx}"] = tensor_to_list(vector)
                
                # Add comprehensive metadata
                emotion_data["metadata"] = {
                    "emotion": emotion,
                    "model": self.model_name,
                    "dtype": str(self.torch_dtype),
                    "quantization": "none",
                    "num_layers": len(emotion_vectors[emotion]),
                    "vector_dimension": len(emotion_vectors[emotion][0]) if emotion_vectors[emotion] else 0,
                    "extraction_method": "direct_from_pretrained_qwen25",
                    "paper_reference": "arXiv:2502.04075v2",
                    "extraction_date": datetime.now().isoformat(),
                    "num_queries_processed": num_queries_per_emotion,
                    "device": self.device,
                    "max_length": self.max_length
                }
                
                # Save to file
                output_file = os.path.join(output_dir, f"Qwen25_1B5_{emotion}_vectors.json")
                with open(output_file, 'w') as f:
                    json.dump(emotion_data, f, indent=2)
                
                file_size = os.path.getsize(output_file) / 1024  # Size in KB
                logger.info(f"Saved {emotion} vectors: {len(emotion_vectors[emotion])} layers ({file_size:.2f} KB)")
        
        # Save base emotion vector
        if base_vector:
            base_data = {}
            
            for layer_idx, vector in enumerate(base_vector):
                base_data[f"layer_{layer_idx}"] = tensor_to_list(vector)
            
            base_data["metadata"] = {
                "emotion": "base_average",
                "model": self.model_name,
                "dtype": str(self.torch_dtype),
                "quantization": "none",
                "num_layers": len(base_vector),
                "vector_dimension": len(base_vector[0]) if base_vector else 0,
                "extraction_method": "direct_from_pretrained_qwen25",
                "paper_reference": "arXiv:2502.04075v2",
                "extraction_date": datetime.now().isoformat(),
                "description": "Average of all emotion vectors across all queries and emotions",
                "device": self.device,
                "max_length": self.max_length
            }
            
            base_file = os.path.join(output_dir, "Qwen25_1B5_base_vectors.json")
            with open(base_file, 'w') as f:
                json.dump(base_data, f, indent=2)
            
            file_size = os.path.getsize(base_file) / 1024  # Size in KB
            logger.info(f"Saved base emotion vector: {len(base_vector)} layers ({file_size:.2f} KB)")
        
        # Create comprehensive extraction summary
        summary_data = {
            "extraction_info": {
                "model_name": self.model_name,
                "model_type": "Qwen2.5-1.5B-Instruct", 
                "dtype": str(self.torch_dtype),
                "quantization": "none",
                "method": "direct_extraction_no_finetuning",
                "paper_reference": "From Rational Answers to Emotional Resonance: The Role of Controllable Emotion Generation in Language Models (arXiv:2502.04075v2)",
                "extraction_date": datetime.now().isoformat(),
                "total_emotions": len(self.emotions),
                "emotions_processed": [emotion for emotion in self.emotions if emotion in emotion_vectors],
                "num_queries_per_emotion": num_queries_per_emotion,
                "successful_extractions": len([e for e in emotion_vectors.values() if e]),
                "device": self.device,
                "max_length": self.max_length
            },
            "model_config": {
                "hidden_layers": self.model.config.num_hidden_layers,
                "hidden_size": self.model.config.hidden_size,
                "vocab_size": self.model.config.vocab_size,
                "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', self.max_length),
                "model_dtype": str(next(self.model.parameters()).dtype)
            },
            "vector_info": {
                "num_layers": len(base_vector) if base_vector else 0,
                "vector_dimension": len(base_vector[0]) if base_vector else 0,
                "emotions": self.emotions,
                "precision": f"{self.torch_dtype}_no_quantization"
            },
            "files_created": {
                "individual_emotions": [f"Qwen25_1B5_{emotion}_vectors.json" for emotion in self.emotions],
                "base_vector": "Qwen25_1B5_base_vectors.json",
                "summary": "extraction_summary.json"
            },
            "methodology": {
                "prompt_template_emotional": "System: You are an AI assistant. Respond with strong {emotion} emotion...",
                "prompt_template_neutral": "System: You are an AI assistant. Respond neutrally and objectively...",
                "chat_formatting": "Qwen chat template with system/user/assistant structure",
                "computation": "emotion_vector = mean_pooled(emotional_hidden_states) - mean_pooled(neutral_hidden_states)",
                "pooling_strategy": "mean-pooled across all tokens in sequence (dim=1)",
                "averaging": "across_queries_per_emotion_and_layers",
                "hidden_state_extraction": "layer-wise mean-pooling during generation or forward pass",
                "paper_methodology": "Following Section 3.1 of arXiv:2502.04075v2"
            }
        }
        
        summary_file = os.path.join(output_dir, "extraction_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        file_size = os.path.getsize(summary_file) / 1024  # Size in KB
        logger.info(f"Saved extraction summary ({file_size:.2f} KB)")

def load_emotion_queries(emotion_queries_file: str) -> Dict[str, List[str]]:
    """
    Load emotion queries from JSON file
    
    Args:
        emotion_queries_file: Path to EmotionQuery.json file
        
    Returns:
        Dictionary mapping emotions to lists of queries
    """
    logger.info(f"Loading emotion queries from: {emotion_queries_file}")
    
    try:
        with open(emotion_queries_file, 'r', encoding='utf-8') as f:
            emotion_queries = json.load(f)
        
        logger.info("Emotion queries loaded successfully:")
        for emotion, queries in emotion_queries.items():
            logger.info(f"  • {emotion.capitalize()}: {len(queries)} queries")
        
        return emotion_queries
        
    except FileNotFoundError:
        logger.error(f"EmotionQuery file not found: {emotion_queries_file}")
        logger.info("Creating fallback emotion queries...")
        
        # Fallback emotion queries for testing
        fallback_queries = {
            "joy": [
                "How do you feel when you achieve something great?",
                "What's your reaction to receiving wonderful news?",
                "Describe your happiness when you succeed at a difficult task.",
                "How do you express joy and excitement?",
                "What makes you feel truly delighted and cheerful?"
            ],
            "anger": [
                "How do you feel when someone treats you unfairly?",
                "What's your reaction to injustice and wrongdoing?",
                "Describe your frustration with dishonest behavior.",
                "How do you express anger and irritation?",
                "What makes you feel furious and outraged?"
            ],
            "disgust": [
                "How do you feel about unethical and immoral actions?",
                "What's your reaction to repulsive behavior?",
                "Describe your revulsion to corrupt practices.",
                "How do you express disgust and repugnance?",
                "What makes you feel sickened and appalled?"
            ],
            "fear": [
                "How do you feel when facing unknown dangers?",
                "What's your reaction to threatening situations?",
                "Describe your anxiety about uncertain outcomes.",
                "How do you express fear and worry?",
                "What makes you feel terrified and anxious?"
            ],
            "sadness": [
                "How do you feel when you lose something important?",
                "What's your reaction to disappointing news?",
                "Describe your grief over significant losses.",
                "How do you express sadness and melancholy?",
                "What makes you feel heartbroken and sorrowful?"
            ]
        }
        
        logger.info("Using fallback emotion queries:")
        for emotion, queries in fallback_queries.items():
            logger.info(f"  • {emotion.capitalize()}: {len(queries)} queries")
        
        return fallback_queries
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading emotion queries: {e}")
        raise

def main():
    """Main function to run emotion vector extraction"""
    
    parser = argparse.ArgumentParser(
        description="Extract emotion vectors from Qwen-2.5-1.5B-Instruct model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--num_queries", 
        type=int, 
        required=True,
        help="Number of queries to use per emotion for extraction"
    )
    parser.add_argument(
        "--emotion_queries_file", 
        type=str, 
        default="EmotionQuery.json",
        help="Path to emotion queries JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./emotion_vectors",
        help="Directory to save extracted emotion vectors"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run model on (cuda, cpu, cuda:0, etc.)"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=2048,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model data type"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (currently not implemented)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="averaged",
        choices=["averaged", "per_prompt", "both"],
        help="'averaged' saves only the query-averaged vectors (original "
             "behaviour). 'per_prompt' saves the per-query hidden states as "
             ".npz for t-SNE. 'both' saves both."
    )
    parser.add_argument(
        "--per_prompt_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="On-disk precision for the per-prompt arrays (float32 doubles size)"
    )
    parser.add_argument(
        "--per_prompt_dir",
        type=str,
        default=None,
        help="Directory for the per-prompt .npz files (default: <output_dir>/per_prompt)"
    )

    args = parser.parse_args()

    if args.per_prompt_dir is None:
        args.per_prompt_dir = os.path.join(args.output_dir, "per_prompt")
    
    # Convert string dtype to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    
    logger.info("="*80)
    logger.info("QWEN-2.5 EMOTION VECTOR EXTRACTION")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Queries per emotion: {args.num_queries}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Emotion queries file: {args.emotion_queries_file}")
    logger.info(f"Mode: {args.mode}")
    if args.mode in ("per_prompt", "both"):
        logger.info(f"Per-prompt directory: {args.per_prompt_dir}")
        logger.info(f"Per-prompt dtype: {args.per_prompt_dtype}")
    logger.info("="*80)
    
    # Check device availability
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load emotion queries
        emotion_queries = load_emotion_queries(args.emotion_queries_file)
        
        # Check if we have enough queries for each emotion
        for emotion, queries in emotion_queries.items():
            if len(queries) < args.num_queries:
                logger.warning(f"Emotion '{emotion}' has only {len(queries)} queries, but {args.num_queries} requested")
                logger.warning(f"Will use all {len(queries)} available queries for '{emotion}'")
        
        # Initialize extractor
        start_time = time.time()
        extractor = QwenEmotionVectorExtractor(
            model_name=args.model_name,
            device=args.device,
            max_length=args.max_length,
            torch_dtype=torch_dtype
        )
        
        init_time = time.time() - start_time
        logger.info(f"Model initialization completed in {init_time:.2f} seconds")

        # Enable per-query capture before extraction starts
        if args.mode in ("per_prompt", "both"):
            extractor.collect_per_prompt = True
            extractor.per_prompt_dtype = (
                np.float16 if args.per_prompt_dtype == "float16" else np.float32
            )
            extractor.per_prompt_dir = args.per_prompt_dir

        # Extract emotion vectors
        logger.info("Starting emotion vector extraction...")
        extraction_start = time.time()
        
        emotion_vectors, base_vector = extractor.extract_emotion_vectors(
            emotion_queries=emotion_queries,
            num_queries_per_emotion=args.num_queries
        )
        
        extraction_time = time.time() - extraction_start
        logger.info(f"Emotion vector extraction completed in {extraction_time:.2f} seconds")
        
        # Save results
        logger.info("Saving emotion vectors...")
        save_start = time.time()

        # The per-prompt .npz files were already flushed per emotion during
        # extraction; only the manifest is left to write here.
        if args.mode in ("per_prompt", "both"):
            extractor.save_per_prompt_manifest(
                output_dir=args.per_prompt_dir,
                num_queries_per_emotion=args.num_queries,
                emotion_queries_file=args.emotion_queries_file
            )

        if args.mode in ("averaged", "both"):
            extractor.save_emotion_vectors(
                emotion_vectors=emotion_vectors,
                base_vector=base_vector,
                output_dir=args.output_dir,
                num_queries_per_emotion=args.num_queries
            )
        else:
            logger.info("Mode 'per_prompt': skipping averaged vector JSON files")

        save_time = time.time() - save_start
        logger.info(f"Saving completed in {save_time:.2f} seconds")
        
        # Print final summary
        total_time = time.time() - start_time
        logger.info("="*80)
        logger.info("EXTRACTION COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Emotions processed: {len(emotion_vectors)}")
        logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
        
        if emotion_vectors:
            first_emotion = next(iter(emotion_vectors.keys()))
            num_layers = len(emotion_vectors[first_emotion])
            vector_dim = len(emotion_vectors[first_emotion][0])
            logger.info(f"Vector properties: {num_layers} layers × {vector_dim} dimensions")
        
        logger.info("Files created:")
        if os.path.exists(args.output_dir):
            for file in sorted(os.listdir(args.output_dir)):
                if file.endswith('.json'):
                    file_path = os.path.join(args.output_dir, file)
                    file_size = os.path.getsize(file_path) / 1024
                    logger.info(f"  • {file} ({file_size:.2f} KB)")

        if args.mode in ("per_prompt", "both") and os.path.exists(args.per_prompt_dir):
            logger.info(f"Per-prompt files ({args.per_prompt_dir}):")
            for file in sorted(os.listdir(args.per_prompt_dir)):
                if file.endswith(('.npz', '.json')):
                    file_path = os.path.join(args.per_prompt_dir, file)
                    file_size = os.path.getsize(file_path) / 1e6
                    logger.info(f"  • {file} ({file_size:.2f} MB)")

        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)
    
    finally:
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")

if __name__ == "__main__":
    main()