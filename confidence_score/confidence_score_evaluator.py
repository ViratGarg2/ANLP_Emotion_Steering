#!/usr/bin/env python3
"""
Emotion Vector Evaluation Script

This script evaluates the effectiveness of extracted emotion vectors by:
1. Loading pre-extracted emotion vectors from JSON files
2. Testing model confidence scores before and after adding emotion vectors
3. Using BART-large-MNLI for emotion classification
4. Comparing classification confidence with configurable alpha multiplier

Usage:
    python emotion_vector_evaluation.py --alpha 2.0 --num_queries 100
    python emotion_vector_evaluation.py --alpha 1.5 --emotion joy --num_queries 100
    python emotion_vector_evaluation.py --alpha 3.0 --output_file results.csv --num_queries 100

Author: Emotion Vector Evaluation System
Date: November 2025
"""

import argparse
import json
import os
import sys
import time
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
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
        AutoModelForSequenceClassification,
        pipeline,
        BartTokenizer,
        BartForSequenceClassification
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
        logging.FileHandler('emotion_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmotionVectorEvaluator:
    """
    Emotion Vector Evaluator for testing EV effectiveness
    
    Evaluates emotion vectors by:
    1. Loading pre-extracted emotion vectors
    2. Injecting vectors into model hidden states with alpha multiplier
    3. Measuring classification confidence before/after injection
    4. Using BART-large-MNLI for emotion classification
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 classifier_name: str = "facebook/bart-large-mnli",
                 device: str = "cuda",
                 torch_dtype: torch.dtype = torch.float32):
        """
        Initialize the Emotion Vector Evaluator
        
        Args:
            model_name: HuggingFace model identifier for generation
            classifier_name: HuggingFace model for emotion classification
            device: Device to run models on
            torch_dtype: PyTorch data type for model weights
        """
        self.model_name = model_name
        self.classifier_name = classifier_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Initialize emotion vector injection variables
        self.hooks = []
        self.current_emotion_vectors = None
        self.current_alpha = 0.0
        self.layer_idx = 0
        
        logger.info(f"Initializing Emotion Vector Evaluator")
        logger.info(f"Generation Model: {model_name}")
        logger.info(f"Classification Model: {classifier_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Data Type: {torch_dtype}")
        
        # Load models and tokenizers
        self._load_generation_model()
        self._load_classification_model()
        
        # Define emotion categories (matching the emotion vector names)
        self.emotions = ["joy", "anger", "disgust", "fear", "sadness"]
        
        # Note: Removed hardcoded emotion_to_labels mapping as BART-MNLI can directly
        # classify against emotion names without requiring synonym expansion
        
        logger.info(f"Evaluator initialized successfully")
        logger.info(f"Target emotions: {self.emotions}")
    
    def __del__(self):
        """Cleanup method to ensure hooks are removed when evaluator is destroyed"""
        if hasattr(self, 'hooks') and self.hooks:
            logger.debug("🧹 Cleaning up emotion vector hooks on evaluator destruction")
            self._remove_emotion_hooks()
    
    def _load_generation_model(self):
        """Load Qwen model and tokenizer for text generation"""
        
        logger.info("Loading generation model...")
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
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device.startswith("cuda") else None,
                trust_remote_code=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                low_cpu_mem_usage=True,
            )
            
            # Move to device if not using device_map
            if not self.device.startswith("cuda") or self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info(f"Generation model loaded successfully!")
            logger.info(f"Parameters: {self.model.num_parameters():,}")
            logger.info(f"Layers: {self.model.config.num_hidden_layers}")
            
        except Exception as e:
            logger.error(f"Failed to load generation model: {e}")
            raise
    
    def _load_classification_model(self):
        """Load BART-large-MNLI model for emotion classification"""
        
        logger.info("Loading classification model...")
        try:
            # Use pipeline for easier MNLI classification
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.classifier_name,
                device=0 if self.device.startswith("cuda") and torch.cuda.is_available() else -1
            )
            
            logger.info(f"Classification model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load classification model: {e}")
            raise
    
    def load_emotion_vectors(self, vectors_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load emotion vectors from JSON files
        
        Args:
            vectors_dir: Directory containing emotion vector JSON files
            
        Returns:
            Dictionary mapping emotions to layer vectors
        """
        logger.info(f"Loading emotion vectors from: {vectors_dir}")
        
        emotion_vectors = {}
        
        for emotion in self.emotions:
            vector_file = os.path.join(vectors_dir, f"Qwen25_1B5_{emotion}_vectors.json")
            
            if os.path.exists(vector_file):
                try:
                    with open(vector_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract layer vectors and calculate normalization statistics
                    layer_vectors = {}
                    vector_norms = []
                    
                    for key, value in data.items():
                        if key.startswith("layer_"):
                            layer_idx = int(key.split("_")[1])
                            vector_tensor = torch.tensor(value, dtype=self.torch_dtype)
                            layer_vectors[layer_idx] = vector_tensor
                            
                            # Calculate norm for statistics
                            vector_norm = torch.norm(vector_tensor).item()
                            vector_norms.append(vector_norm)
                    
                    emotion_vectors[emotion] = layer_vectors
                    
                    # Log vector statistics
                    if vector_norms:
                        avg_norm = np.mean(vector_norms)
                        std_norm = np.std(vector_norms)
                        min_norm = np.min(vector_norms)
                        max_norm = np.max(vector_norms)
                        
                        logger.info(f"Loaded {emotion}: {len(layer_vectors)} layers")
                        logger.info(f"  📊 Vector norms - Avg: {avg_norm:.3f}, Std: {std_norm:.3f}, Range: [{min_norm:.3f}, {max_norm:.3f}]")
                        logger.info(f"  🔧 Vectors will be normalized to unit length during injection")
                    else:
                        logger.info(f"Loaded {emotion}: {len(layer_vectors)} layers")
                    
                except Exception as e:
                    logger.error(f"Failed to load {emotion} vectors: {e}")
                    continue
            else:
                logger.warning(f"Vector file not found: {vector_file}")
        
        logger.info(f"Loaded emotion vectors for {len(emotion_vectors)} emotions")
        logger.info("🔧 All emotion vectors will be L2-normalized to unit length during injection")
        
        # Validate vector loading and provide normalization summary
        total_vectors = sum(len(layer_vecs) for layer_vecs in emotion_vectors.values())
        logger.info(f"📈 Total vectors loaded: {total_vectors} across {len(emotion_vectors)} emotions")
        
        return emotion_vectors
    
    def _normalize_emotion_vector(self, emotion_vector: torch.Tensor, emotion_name: str = "", layer_num: int = -1) -> torch.Tensor:
        """
        Normalize emotion vector to unit length with detailed logging
        
        Args:
            emotion_vector: Raw emotion vector tensor
            emotion_name: Name of the emotion for logging (optional)
            layer_num: Layer number for logging (optional)
            
        Returns:
            L2-normalized emotion vector with unit norm (||v|| = 1.0)
        """
        # Check for zero vectors (would cause division by zero in normalization)
        if torch.norm(emotion_vector).item() < 1e-8:
            logger.warning(f"⚠️  Zero or near-zero emotion vector detected for {emotion_name} layer {layer_num}")
            return emotion_vector
        
        original_norm = torch.norm(emotion_vector).item()
        normalized_vector = torch.nn.functional.normalize(emotion_vector, p=2, dim=0)
        new_norm = torch.norm(normalized_vector).item()
        
        # Verify normalization worked correctly
        if abs(new_norm - 1.0) > 1e-6:
            logger.warning(f"⚠️  Normalization check failed: expected 1.0, got {new_norm:.6f}")
        
        if emotion_name and layer_num >= 0:
            logger.debug(f"  🔧 Normalized {emotion_name} vector at layer {layer_num}: {original_norm:.4f} → {new_norm:.4f}")
        
        return normalized_vector
    
    def _format_qwen_prompt(self, system_message: str, user_message: str) -> str:
        """Format prompt using Qwen's chat template"""
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
        
    def _create_emotion_injection_hook(self, layer_name: str):
        """
        Create a forward hook for emotion vector injection
        Based on the reference implementation from xuanfengzu/EmotionVector
        """
        def hook_function(module, input_tensor, output_tensor):
            if self.current_emotion_vectors is not None and self.current_alpha != 0:
                # Check if this is a transformer layer we want to modify
                if "layers" in layer_name or "layer" in layer_name.lower():
                    # Get the layer index from the layer name
                    try:
                        # Extract layer number from layer name (e.g., "model.layers.15" -> 15)
                        if "layers." in layer_name:
                            layer_num = int(layer_name.split("layers.")[1].split(".")[0])
                        elif "layer" in layer_name.lower():
                            # Alternative layer naming convention
                            import re
                            match = re.search(r'layer[._]?(\d+)', layer_name.lower())
                            layer_num = int(match.group(1)) if match else self.layer_idx
                        else:
                            layer_num = self.layer_idx
                        
                        # Check if we have emotion vector for this layer
                        if layer_num in self.current_emotion_vectors:
                            emotion_vector = self.current_emotion_vectors[layer_num]
                            
                            # Normalize the emotion vector to unit length
                            # This ensures consistent magnitude across different emotions and layers
                            original_norm = torch.norm(emotion_vector).item()
                            emotion_vector_norm = torch.nn.functional.normalize(emotion_vector, p=2, dim=0)
                            
                            # Apply emotion vector to the last token's hidden state
                            # Following the reference implementation pattern
                            if isinstance(output_tensor, tuple):
                                hidden_states = output_tensor[0]
                            else:
                                hidden_states = output_tensor
                            
                            # Inject normalized emotion vector with alpha multiplier to the last token
                            # Shape: [batch_size, seq_len, hidden_size]
                            if len(hidden_states.shape) == 3:
                                # Add normalized emotion vector to the last token position
                                hidden_states[:, -1:, :] = (
                                    hidden_states[:, -1:, :] + 
                                    self.current_alpha * emotion_vector_norm.unsqueeze(0).unsqueeze(0).to(hidden_states.device)
                                )
                                
                                logger.debug(f"🎯 NORMALIZED EV INJECTED at layer {layer_num}: α={self.current_alpha:.2f}, norm {original_norm:.3f}→1.000")
                            
                            # Return modified output
                            if isinstance(output_tensor, tuple):
                                return (hidden_states,) + output_tensor[1:]
                            else:
                                return hidden_states
                        
                        # Increment layer index for tracking
                        self.layer_idx += 1
                        
                    except (ValueError, IndexError, AttributeError) as e:
                        logger.debug(f"Could not extract layer number from {layer_name}: {e}")
            
            return output_tensor
        
        return hook_function
    
    def _register_emotion_hooks(self):
        """Register forward hooks on transformer layers for emotion vector injection"""
        logger.debug("🔗 Registering emotion vector injection hooks...")
        self.layer_idx = 0
        
        for name, module in self.model.named_modules():
            # Register hooks on transformer layers
            if ("layers" in name and "mlp" not in name and "attn" not in name and 
                name.count('.') <= 3):  # Target main layer modules
                hook = module.register_forward_hook(
                    self._create_emotion_injection_hook(name)
                )
                self.hooks.append(hook)
                logger.debug(f"  📎 Hook registered on: {name}")
    
    def _remove_emotion_hooks(self):
        """Remove all registered forward hooks"""
        logger.debug("🔓 Removing emotion vector injection hooks...")
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_idx = 0

    def generate_with_emotion_vector(self, 
                                   prompt: str, 
                                   emotion_vectors: Optional[Dict[int, torch.Tensor]] = None,
                                   alpha: float = 1.0,
                                   max_new_tokens: int = 50) -> str:
        """
        Generate text with emotion vector injection via model hooks
        
        Args:
            prompt: Input prompt for generation
            emotion_vectors: Dictionary of layer_idx -> emotion_vector
            alpha: Multiplier for emotion vector strength
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=2048,
                padding=False
            ).to(self.device)
            
            # Set up emotion vector injection if provided
            if emotion_vectors and alpha != 0:
                # Calculate normalization statistics for logging
                vector_norms = [torch.norm(vec).item() for vec in emotion_vectors.values()]
                avg_norm = np.mean(vector_norms)
                std_norm = np.std(vector_norms)
                
                logger.info("🚀 USING EMOTION VECTOR INJECTION FOR TEXT GENERATION")
                logger.info(f"   📊 Alpha multiplier: {alpha}")
                logger.info(f"   🎭 Emotion vectors: {len(emotion_vectors)} layers")
                logger.info(f"   📏 Vector norms - Avg: {avg_norm:.3f} ± {std_norm:.3f} (will be normalized to 1.000)")
                logger.info(f"   ⚡ Injection method: Hidden state modification via forward hooks with L2 normalization")
                
                # Set current emotion vectors for hook access
                self.current_emotion_vectors = emotion_vectors
                self.current_alpha = alpha
                
                # Register hooks for emotion vector injection
                self._register_emotion_hooks()
                
            else:
                logger.info("🔍 Generating baseline text WITHOUT emotion vectors")
                self.current_emotion_vectors = None
                self.current_alpha = 0.0
            
            # Generate response with emotion vector injection
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Clean up hooks after generation
            if emotion_vectors and alpha != 0:
                self._remove_emotion_hooks()
                logger.info("✅ Emotion vector injection completed and hooks removed")
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            # Make sure to clean up hooks on error
            if self.hooks:
                self._remove_emotion_hooks()
            return ""
    
    def classify_emotion_confidence(self, text: str, target_emotion: str) -> Dict[str, float]:
        """
        Classify emotion in text and return confidence scores using BART-MNLI
        
        Args:
            text: Text to classify
            target_emotion: Target emotion to evaluate (should match EV emotion names)
            
        Returns:
            Dictionary with emotion labels and confidence scores
        """
        try:
            # Use all emotion categories as candidate labels for comprehensive comparison
            # This includes the target emotion and other emotions for relative scoring
            candidate_labels = ["joy", "anger", "disgust", "fear", "sadness", "neutral"]
            
            # Classify using BART-MNLI zero-shot classification
            # BART-MNLI treats each label as a hypothesis and scores likelihood
            result = self.classifier(text, candidate_labels)
            
            # Convert to confidence dictionary (sorted by confidence score)
            confidence_scores = {}
            for label, score in zip(result['labels'], result['scores']):
                confidence_scores[label] = float(score)
            
            # Log the classification for debugging with full emotion breakdown
            all_scores = ", ".join([f"{label}: {confidence_scores.get(label, 0.0):.3f}" 
                                   for label in ["joy", "anger", "disgust", "fear", "sadness", "neutral"]])
            logger.debug(f"Classification for '{text[:30]}...': [{all_scores}] -> Target {target_emotion}: {confidence_scores.get(target_emotion, 0.0):.3f}")
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Error in emotion classification: {e}")
            return {}
    
    def evaluate_emotion_vectors_layerwise(self, 
                                          emotion_vectors: Dict[str, Dict[int, torch.Tensor]],
                                          all_queries: List[str],
                                          alpha: float,
                                          num_queries: int = 25,
                                          target_emotion: str = "joy") -> List[Dict[str, Any]]:
        """
        Evaluate joy emotion vectors layer by layer on first 25 queries from EmotionQuery+
        
        Args:
            emotion_vectors: Loaded emotion vectors
            all_queries: List of all queries from EmotionQuery+.json
            alpha: Multiplier for emotion vector strength
            num_queries: Number of queries to test (default: 25)
            target_emotion: Emotion to test (default: joy)
            
        Returns:
            List of evaluation results for each layer
        """
        logger.info("Starting layerwise emotion vector evaluation...")
        logger.info(f"Target emotion: {target_emotion}")
        logger.info(f"Alpha multiplier: {alpha}")
        logger.info(f"Total queries available: {len(all_queries)}")
        logger.info(f"Queries to evaluate: {num_queries} (first {num_queries} from EmotionQuery+)")
        logger.info(f"Approach: Testing each layer individually for {target_emotion}")
        
        # Check if we have vectors for the target emotion
        if target_emotion not in emotion_vectors:
            logger.error(f"No vectors found for emotion: {target_emotion}")
            return []
        
        emotion_vector_layers = emotion_vectors[target_emotion]
        logger.info(f"Available layers for {target_emotion}: {sorted(emotion_vector_layers.keys())}")
        
        # Take first num_queries from EmotionQuery+ dataset
        evaluation_queries = all_queries[:num_queries]
        logger.info(f"✅ Using first {len(evaluation_queries)} queries from EmotionQuery+.json")
        
        all_results = []
        
        # Test each layer individually
        for layer_num in sorted(emotion_vector_layers.keys()):
            logger.info(f"Evaluating layer {layer_num}/{max(emotion_vector_layers.keys())}")
            
            # Create single-layer emotion vectors dict for this layer only
            single_layer_vectors = {layer_num: emotion_vector_layers[layer_num]}
            
            for query_idx, query in enumerate(tqdm(evaluation_queries, desc=f"Layer-{layer_num}")):
                try:
                    # Create neutral system message
                    neutral_system = (
                        "You are an AI assistant. Respond neutrally and objectively to the user's query "
                        "without showing any particular emotion. Maintain a balanced and factual tone."
                    )
                    
                    # Format prompt
                    prompt = self._format_qwen_prompt(neutral_system, query)
                    
                    # Generate baseline response (without emotion vector)
                    baseline_response = self.generate_with_emotion_vector(
                        prompt=prompt,
                        emotion_vectors=None,
                        alpha=0.0
                    )
                    
                    # Generate response WITH emotion vector injection for THIS layer only
                    enhanced_response = self.generate_with_emotion_vector(
                        prompt=prompt,
                        emotion_vectors=single_layer_vectors,  # Only this layer
                        alpha=alpha
                    )
                    
                    # Classify both responses for the target emotion
                    baseline_confidence = self.classify_emotion_confidence(baseline_response, target_emotion)
                    enhanced_confidence = self.classify_emotion_confidence(enhanced_response, target_emotion)
                    
                    # Extract confidence scores for the target emotion
                    baseline_target_score = baseline_confidence.get(target_emotion, 0.0)
                    enhanced_target_score = enhanced_confidence.get(target_emotion, 0.0)
                    
                    # Calculate improvement in target emotion confidence
                    confidence_change = enhanced_target_score - baseline_target_score
                    
                    # Store results
                    result = {
                        "emotion": target_emotion,
                        "layer": layer_num,
                        "query_idx": query_idx,
                        "query": query,
                        "alpha": alpha,
                        "baseline_response": baseline_response,
                        "enhanced_response": enhanced_response,
                        "baseline_confidence_all": baseline_confidence,
                        "enhanced_confidence_all": enhanced_confidence,
                        "baseline_target_confidence": baseline_target_score,
                        "enhanced_target_confidence": enhanced_target_score,
                        "confidence_change": confidence_change,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    all_results.append(result)
                    
                    logger.debug(f"Layer {layer_num}, Query {query_idx+1}: {confidence_change:+.3f} confidence change")
                    
                except Exception as e:
                    logger.error(f"Error evaluating layer {layer_num}, query {query_idx+1}: {e}")
                    continue
        
        # Log final evaluation summary
        logger.info("="*50)
        logger.info("LAYERWISE EVALUATION SUMMARY:")
        logger.info(f"Emotion tested: {target_emotion}")
        logger.info(f"Total evaluations: {len(all_results)}")
        logger.info(f"Layers tested: {len(set([r['layer'] for r in all_results]))}")
        logger.info(f"Queries per layer: {num_queries}")
        logger.info("="*50)
        
        return all_results

    def evaluate_emotion_vectors(self,
                                 emotion_vectors: Dict[str, Dict[int, torch.Tensor]],
                                 all_queries: List[str],
                                 alpha: float,
                                 num_queries_per_emotion: int = 25,
                                 target_emotions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate emotion vectors injected at all layers simultaneously

        Args:
            emotion_vectors: Loaded emotion vectors
            all_queries: List of all queries
            alpha: Multiplier for emotion vector strength
            num_queries_per_emotion: Number of queries per emotion
            target_emotions: Emotions to test

        Returns:
            List of evaluation results
        """
        logger.info("Starting regular emotion vector evaluation...")

        emotions_to_test = target_emotions if target_emotions else self.emotions
        evaluation_queries = all_queries[:num_queries_per_emotion]

        all_results = []
        emotions_evaluated = []

        for emotion in emotions_to_test:
            if emotion not in emotion_vectors:
                logger.warning(f"No vectors found for emotion: {emotion}")
                continue

            emotions_evaluated.append(emotion)
            emotion_vector_layers = emotion_vectors[emotion]

            for query_idx, query in enumerate(tqdm(evaluation_queries, desc=f"Evaluating {emotion}")):
                try:
                    neutral_system = (
                        "You are an AI assistant. Respond neutrally and objectively to the user's query "
                        "without showing any particular emotion. Maintain a balanced and factual tone."
                    )

                    prompt = self._format_qwen_prompt(neutral_system, query)

                    baseline_response = self.generate_with_emotion_vector(
                        prompt=prompt,
                        emotion_vectors=None,
                        alpha=0.0
                    )

                    enhanced_response = self.generate_with_emotion_vector(
                        prompt=prompt,
                        emotion_vectors=emotion_vector_layers,
                        alpha=alpha
                    )

                    baseline_confidence = self.classify_emotion_confidence(baseline_response, emotion)
                    enhanced_confidence = self.classify_emotion_confidence(enhanced_response, emotion)

                    baseline_target_score = baseline_confidence.get(emotion, 0.0)
                    enhanced_target_score = enhanced_confidence.get(emotion, 0.0)

                    confidence_change = enhanced_target_score - baseline_target_score

                    result = {
                        "emotion": emotion,
                        "layer": "ALL_LAYERS",
                        "query_idx": query_idx,
                        "query": query,
                        "alpha": alpha,
                        "baseline_response": baseline_response,
                        "enhanced_response": enhanced_response,
                        "baseline_max_confidence": baseline_target_score,
                        "enhanced_max_confidence": enhanced_target_score,
                        "confidence_improvement": confidence_change,
                        "timestamp": datetime.now().isoformat()
                    }

                    all_results.append(result)

                    logger.debug(f"{emotion}, Query {query_idx+1}: {confidence_change:+.3f} confidence change")

                except Exception as e:
                    logger.error(f"Error evaluating emotion {emotion}, query {query_idx+1}: {e}")
                    continue

        # Log final evaluation summary
        logger.info("="*50)
        logger.info("EVALUATION SUMMARY:")
        for emotion in emotions_evaluated:
            emotion_results = [r for r in all_results if r["emotion"] == emotion]
            logger.info(f"  • {emotion.upper()}: {len(emotion_results)} queries evaluated")

        logger.info(f"TOTAL: {len(all_results)} evaluations across {len(emotions_evaluated)} emotions")
        logger.info("="*50)

        return all_results

    def save_layerwise_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save layerwise evaluation results to CSV file
        
        Args:
            results: List of evaluation results
            output_file: Output CSV file path
        """
        if not results:
            logger.warning("No results to save")
            return
        
        logger.info(f"Saving {len(results)} layerwise evaluation results to: {output_file}")
        
        # Define CSV columns for layerwise analysis
        fieldnames = [
            "emotion", "layer", "query_idx", "alpha", 
            "baseline_target_confidence", "enhanced_target_confidence", "confidence_change",
            "query", "baseline_response", "enhanced_response", 
            "baseline_confidence_all", "enhanced_confidence_all", "timestamp"
        ]
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Flatten the result for CSV writing
                    csv_row = {
                        "emotion": result["emotion"],
                        "layer": result["layer"],
                        "query_idx": result["query_idx"],
                        "alpha": result["alpha"],
                        "baseline_target_confidence": result["baseline_target_confidence"],
                        "enhanced_target_confidence": result["enhanced_target_confidence"],
                        "confidence_change": result["confidence_change"],
                        "query": result["query"],
                        "baseline_response": result["baseline_response"],
                        "enhanced_response": result["enhanced_response"],
                        "baseline_confidence_all": json.dumps(result["baseline_confidence_all"]),
                        "enhanced_confidence_all": json.dumps(result["enhanced_confidence_all"]),
                        "timestamp": result["timestamp"]
                    }
                    writer.writerow(csv_row)
            
            file_size = os.path.getsize(output_file) / 1024  # Size in KB
            logger.info(f"Layerwise results saved successfully ({file_size:.2f} KB)")
            
        except Exception as e:
            logger.error(f"Error saving layerwise results: {e}")
            raise

    def save_layerwise_summary(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save layerwise summary statistics to JSON file
        
        Args:
            results: List of evaluation results
            output_file: Output JSON file path for summary
        """
        if not results:
            logger.warning("No results to summarize")
            return
        
        logger.info("Generating layerwise summary statistics...")
        
        # Group results by layer
        layer_stats = {}
        emotion = results[0]["emotion"]  # Should all be the same emotion
        
        for layer_num in sorted(set([r["layer"] for r in results])):
            layer_results = [r for r in results if r["layer"] == layer_num]
            
            # Calculate statistics for this layer
            confidence_changes = [r["confidence_change"] for r in layer_results]
            baseline_scores = [r["baseline_target_confidence"] for r in layer_results]
            enhanced_scores = [r["enhanced_target_confidence"] for r in layer_results]
            
            positive_changes = [change for change in confidence_changes if change > 0]
            negative_changes = [change for change in confidence_changes if change < 0]
            zero_changes = [change for change in confidence_changes if change == 0]
            
            layer_stats[f"layer_{layer_num}"] = {
                "layer_number": layer_num,
                "total_queries": len(layer_results),
                "avg_confidence_change": float(np.mean(confidence_changes)),
                "median_confidence_change": float(np.median(confidence_changes)),
                "std_confidence_change": float(np.std(confidence_changes)),
                "min_confidence_change": float(np.min(confidence_changes)),
                "max_confidence_change": float(np.max(confidence_changes)),
                "positive_changes": len(positive_changes),
                "negative_changes": len(negative_changes),
                "zero_changes": len(zero_changes),
                "positive_rate": float(len(positive_changes) / len(confidence_changes)) if confidence_changes else 0.0,
                "avg_baseline_confidence": float(np.mean(baseline_scores)),
                "avg_enhanced_confidence": float(np.mean(enhanced_scores)),
                "effect_strength": float(np.mean([abs(change) for change in confidence_changes]))
            }
        
        # Overall summary across all layers
        all_confidence_changes = [r["confidence_change"] for r in results]
        overall_summary = {
            "emotion_tested": emotion,
            "total_layers_tested": len(layer_stats),
            "total_evaluations": len(results),
            "queries_per_layer": len(results) // len(layer_stats) if layer_stats else 0,
            "overall_avg_change": float(np.mean(all_confidence_changes)),
            "overall_std_change": float(np.std(all_confidence_changes)),
            "best_layer": max(layer_stats.keys(), key=lambda k: layer_stats[k]["avg_confidence_change"]) if layer_stats else None,
            "worst_layer": min(layer_stats.keys(), key=lambda k: layer_stats[k]["avg_confidence_change"]) if layer_stats else None,
            "most_effective_layer": max(layer_stats.keys(), key=lambda k: layer_stats[k]["effect_strength"]) if layer_stats else None
        }
        
        # Create final summary object
        summary = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "emotion": emotion,
                "alpha": results[0]["alpha"] if results else None,
                "evaluation_type": "layerwise_confidence_analysis"
            },
            "overall_summary": overall_summary,
            "layer_statistics": layer_stats,
            "methodology": {
                "approach": "Individual layer testing for joy emotion",
                "confidence_metric": "BART-large-MNLI classification confidence",
                "queries_source": "First 25 from EmotionQuery+.json",
                "calculation": "confidence_change = enhanced_confidence - baseline_confidence"
            }
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Layerwise summary saved to: {output_file}")
            
            # Log key findings
            if layer_stats:
                best_layer = overall_summary["best_layer"]
                worst_layer = overall_summary["worst_layer"]
                most_effective = overall_summary["most_effective_layer"]
                
                logger.info(f"📊 LAYERWISE ANALYSIS SUMMARY:")
                logger.info(f"   Best performing layer: {best_layer} (avg change: {layer_stats[best_layer]['avg_confidence_change']:+.3f})")
                logger.info(f"   Worst performing layer: {worst_layer} (avg change: {layer_stats[worst_layer]['avg_confidence_change']:+.3f})")
                logger.info(f"   Most effective layer: {most_effective} (effect strength: {layer_stats[most_effective]['effect_strength']:.3f})")
                
        except Exception as e:
            logger.error(f"Error saving layerwise summary: {e}")
            raise

    def save_evaluation_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save evaluation results to CSV file
        
        Args:
            results: List of evaluation results
            output_file: Output CSV file path
        """
        if not results:
            logger.warning("No results to save")
            return
        
        logger.info(f"Saving {len(results)} evaluation results to: {output_file}")
        
        # Define CSV columns
        fieldnames = [
            "emotion", "layer", "query_idx", "alpha", 
            "baseline_max_confidence", "enhanced_max_confidence", "confidence_improvement",
            "query", "baseline_response", "enhanced_response", "timestamp"
        ]
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Flatten the result for CSV writing
                    csv_row = {
                        "emotion": result["emotion"],
                        "layer": result["layer"],
                        "query_idx": result["query_idx"],
                        "alpha": result["alpha"],
                        "baseline_max_confidence": result["baseline_max_confidence"],
                        "enhanced_max_confidence": result["enhanced_max_confidence"],
                        "confidence_improvement": result["confidence_improvement"],
                        "query": result["query"],
                        "baseline_response": result["baseline_response"],
                        "enhanced_response": result["enhanced_response"],
                        "timestamp": result["timestamp"]
                    }
                    writer.writerow(csv_row)
            
            file_size = os.path.getsize(output_file) / 1024  # Size in KB
            logger.info(f"Results saved successfully ({file_size:.2f} KB)")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def save_emotion_summary(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save emotion-specific summary statistics to JSON file
        
        Args:
            results: List of evaluation results
            output_file: Output JSON file path for summary
        """
        if not results:
            logger.warning("No results to summarize")
            return
        
        logger.info("Generating emotion-specific summary statistics...")
        
        # Group results by emotion
        emotion_stats = {}
        
        for emotion in self.emotions:
            emotion_results = [r for r in results if r["emotion"] == emotion]
            
            if not emotion_results:
                continue
            
            # Calculate statistics
            improvements = [r["confidence_improvement"] for r in emotion_results]
            baseline_scores = [r["baseline_max_confidence"] for r in emotion_results]
            enhanced_scores = [r["enhanced_max_confidence"] for r in emotion_results]
            
            positive_improvements = [imp for imp in improvements if imp > 0]
            negative_improvements = [imp for imp in improvements if imp < 0]
            
            # All-layers analysis (simplified since we're using all layers together)
            all_layers_stats = {
                "improvements": improvements,
                "baseline_scores": baseline_scores,
                "enhanced_scores": enhanced_scores,
                "avg_improvement": float(np.mean(improvements)),
                "avg_baseline": float(np.mean(baseline_scores)),
                "avg_enhanced": float(np.mean(enhanced_scores)),
                "success_rate": float(len(positive_improvements) / len(improvements)),
                "num_queries": len(improvements)
            }
            
            emotion_stats[emotion] = {
                "total_queries": len(emotion_results),
                "average_improvement": float(np.mean(improvements)),
                "median_improvement": float(np.median(improvements)),
                "std_improvement": float(np.std(improvements)),
                "min_improvement": float(np.min(improvements)),
                "max_improvement": float(np.max(improvements)),
                "positive_improvements": len(positive_improvements),
                "negative_improvements": len(negative_improvements),
                "success_rate": float(len(positive_improvements) / len(improvements)),
                "average_baseline_confidence": float(np.mean(baseline_scores)),
                "average_enhanced_confidence": float(np.mean(enhanced_scores)),
                "confidence_boost": float(np.mean(enhanced_scores) - np.mean(baseline_scores)),
                "all_layers_analysis": all_layers_stats,
                "evaluation_approach": "ALL_LAYERS_COMBINED"
            }
        
        # Overall summary
        all_improvements = [r["confidence_improvement"] for r in results]
        overall_summary = {
            "total_evaluations": len(results),
            "emotions_tested": list(emotion_stats.keys()),
            "overall_avg_improvement": float(np.mean(all_improvements)),
            "overall_success_rate": float(len([x for x in all_improvements if x > 0]) / len(all_improvements)),
            "best_performing_emotion": max(emotion_stats.items(), key=lambda x: x[1]["average_improvement"])[0] if emotion_stats else None,
            "worst_performing_emotion": min(emotion_stats.items(), key=lambda x: x[1]["average_improvement"])[0] if emotion_stats else None,
            "alpha_multiplier": results[0]["alpha"] if results else None,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        # Combine all statistics
        summary_data = {
            "overall_summary": overall_summary,
            "emotion_statistics": emotion_stats,
            "evaluation_metadata": {
                "model_name": self.model_name,
                "classifier_name": self.classifier_name,
                "device": self.device,
                "total_queries_processed": len(results),
                "emotions_evaluated": len(emotion_stats)
            }
        }
        
        # Save to JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(output_file) / 1024  # Size in KB
            logger.info(f"Emotion summary saved to: {output_file} ({file_size:.2f} KB)")
            
            # Log key findings
            logger.info("=== EMOTION SUMMARY HIGHLIGHTS ===")
            for emotion, stats in emotion_stats.items():
                logger.info(f"{emotion.upper()}:")
                logger.info(f"  • Success Rate: {stats['success_rate']:.1%}")
                logger.info(f"  • Avg Improvement: {stats['average_improvement']:+.4f}")
                logger.info(f"  • Approach: {stats['evaluation_approach']}")
                logger.info(f"  • Queries Tested: {stats['total_queries']}")
            
            logger.info(f"OVERALL: {overall_summary['overall_success_rate']:.1%} success rate")
            logger.info(f"BEST EMOTION: {overall_summary['best_performing_emotion']}")
            
        except Exception as e:
            logger.error(f"Error saving emotion summary: {e}")
            raise

def load_emotion_queries(emotion_queries_file: str) -> List[str]:
    """
    Load emotion queries from JSON file
    
    Args:
        emotion_queries_file: Path to EmotionQuery+.json file (flat list of 400 queries)
        
    Returns:
        List of all queries to be used for all emotions
    """
    logger.info(f"Loading emotion queries from: {emotion_queries_file}")
    
    try:
        with open(emotion_queries_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if data is a flat list (expected format for EmotionQuery+.json)
        if isinstance(data, list):
            logger.info(f"✅ Loaded {len(data)} queries from flat list format")
            logger.info("📋 These queries will be used for ALL emotion evaluations (same queries for fair comparison)")
            return data
        
        elif isinstance(data, dict):
            # If organized by emotion, flatten to single list
            logger.info("Converting organized format to flat list for consistent evaluation...")
            all_queries = []
            for emotion, queries in data.items():
                if isinstance(queries, list):
                    all_queries.extend(queries)
                    logger.info(f"  • Added {len(queries)} queries from {emotion}")
            
            logger.info(f"✅ Total queries collected: {len(all_queries)}")
            logger.info("📋 All queries will be used for ALL emotion evaluations")
            return all_queries
            
        else:
            raise ValueError("Invalid JSON format: expected list or dict")
        
    except FileNotFoundError:
        logger.error(f"EmotionQuery file not found: {emotion_queries_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading emotion queries: {e}")
        raise
        
    except FileNotFoundError:
        logger.error(f"EmotionQuery file not found: {emotion_queries_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading emotion queries: {e}")
        raise

def main():
    """Main function to run emotion vector evaluation"""
    
    parser = argparse.ArgumentParser(
        description="Evaluate emotion vectors using BART-large-MNLI classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--alpha", 
        type=float, 
        required=True,
        help="Multiplier for emotion vector strength (e.g., 1.0, 2.0, 3.0)"
    )
    parser.add_argument(
        "--vectors_dir", 
        type=str, 
        default="./emotion_vectors",
        help="Directory containing emotion vector JSON files"
    )
    parser.add_argument(
        "--emotion_queries_file", 
        type=str, 
        default="EmotionQuery+.json",
        help="Path to emotion queries JSON file"
    )
    parser.add_argument(
        "--num_queries", 
        type=int, 
        default=25,
        help="Number of queries to test (default: 25 for layerwise analysis)"
    )
    parser.add_argument(
        "--layerwise", 
        action="store_true",
        help="Enable layerwise analysis (test each layer individually for joy emotion)"
    )
    parser.add_argument(
        "--target_emotion", 
        type=str, 
        default="joy",
        choices=["joy", "anger", "disgust", "fear", "sadness"],
        help="Target emotion for layerwise analysis (default: joy)"
    )
    parser.add_argument(
        "--max_eval_queries", 
        type=int, 
        default=None,
        help="Maximum total queries to evaluate across all emotions (None for no limit)"
    )
    parser.add_argument(
        "--emotion", 
        type=str, 
        choices=["joy", "anger", "disgust", "fear", "sadness", "all"],
        default="all",
        help="Specific emotion to evaluate (default: all) - ignored in layerwise mode"
    )
    # Removed --layer argument as we now use all layers together
    # parser.add_argument(
    #     "--layer", 
    #     type=int, 
    #     default=None,
    #     help="Specific layer to evaluate (default: all layers)"
    # )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="Output CSV file for results (default: auto-generated)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name for generation"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run models on (cuda, cpu, cuda:0, etc.)"
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.layerwise:
            args.output_file = f"layerwise_confidence_alpha{args.alpha}_{args.target_emotion}_{timestamp}.csv"
        else:
            emotion_suffix = f"_{args.emotion}" if args.emotion != "all" else ""
            args.output_file = f"emotion_evaluation_alpha{args.alpha}{emotion_suffix}_ALL_LAYERS_{timestamp}.csv"
    
    # Generate summary filename
    summary_file = args.output_file.replace('.csv', '_summary.json')
    
    logger.info("="*80)
    if args.layerwise:
        logger.info("LAYERWISE EMOTION VECTOR EVALUATION")
    else:
        logger.info("EMOTION VECTOR EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Alpha multiplier: {args.alpha}")
    
    if args.layerwise:
        logger.info(f"Mode: LAYERWISE ANALYSIS")
        logger.info(f"Target emotion: {args.target_emotion}")
        logger.info(f"Queries to test: {args.num_queries} (first {args.num_queries} from EmotionQuery+)")
        logger.info(f"Approach: Test each layer individually")
    else:
        logger.info(f"Queries per emotion: {args.num_queries}")
        logger.info(f"Max total eval queries: {args.max_eval_queries if args.max_eval_queries else 'No limit'}")
        logger.info(f"Target emotion: {args.emotion}")
        logger.info(f"Layer approach: ALL_LAYERS_COMBINED")
    
    logger.info(f"EV Injection method: HIDDEN_STATE_MANIPULATION via forward hooks")
    logger.info(f"Vectors directory: {args.vectors_dir}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Summary file: {summary_file}")
    logger.info("="*80)
    
    # Check device availability
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load emotion queries (now returns a flat list)
        all_queries = load_emotion_queries(args.emotion_queries_file)
        
        if args.layerwise:
            # Layerwise evaluation mode
            logger.info("📊 LAYERWISE EVALUATION SETUP:")
            logger.info(f"  📋 Total queries loaded: {len(all_queries)}")
            logger.info(f"  🎯 Will use first {args.num_queries} queries")
            logger.info(f"  🧠 Target emotion: {args.target_emotion}")
            logger.info(f"  🔀 Testing each layer individually")
            
            if len(all_queries) < args.num_queries:
                logger.warning(f"  ⚠️  Only {len(all_queries)} queries available (requested {args.num_queries})")
                args.num_queries = len(all_queries)
                logger.info(f"  📋 Adjusted to use all {args.num_queries} available queries")
        else:
            # Standard evaluation mode (existing logic)
            logger.info("📊 QUERY AVAILABILITY ANALYSIS:")
            logger.info(f"  📋 Total queries loaded: {len(all_queries)}")
            logger.info(f"  🎯 Queries needed per emotion: {args.num_queries}")
            
            if len(all_queries) >= args.num_queries:
                logger.info(f"  ✅ Sufficient queries available for sampling")
                logger.info(f"  📊 Will randomly sample {args.num_queries} queries for ALL emotions")
            else:
                logger.warning(f"  ⚠️  Only {len(all_queries)} queries available (need {args.num_queries})")
                logger.info(f"  📋 Will use sampling with replacement to get {args.num_queries} queries")
            
            logger.info("🎯 EVALUATION STRATEGY:")
            logger.info("  • Same queries will be used for ALL emotions")
            logger.info("  • Fair comparison across all emotion vectors")
            logger.info("  • No bias from emotion-specific query selection")
        
        # Note: We don't need to filter queries by emotion anymore since we use the same queries for all emotions
        # The emotion filtering will happen in the evaluation loop based on available emotion vectors
        
        if not args.layerwise:
            # Check if max_eval_queries conflicts with num_queries_per_emotion setting (only for standard mode)
            target_emotions = ["joy", "anger", "disgust", "fear", "sadness"]
            if args.emotion != "all":
                target_emotions = [args.emotion] if args.emotion in target_emotions else []
            
            if args.max_eval_queries:
                expected_total = len(target_emotions) * args.num_queries
                if args.max_eval_queries < expected_total:
                    logger.warning(f"⚠️  max_eval_queries ({args.max_eval_queries}) is less than expected total ({expected_total})")
                    logger.warning(f"    Expected: {args.num_queries} queries × {len(target_emotions)} emotions = {expected_total}")
                    logger.warning(f"    The --num_queries setting will take precedence to ensure equal evaluation per emotion")
                    logger.info(f"📋 Ignoring max_eval_queries to maintain {args.num_queries} queries per emotion")
                else:
                    logger.info(f"✅ max_eval_queries ({args.max_eval_queries}) is sufficient for {args.num_queries} queries per emotion")
            
            # Final evaluation plan summary
            expected_total_queries = len(target_emotions) * args.num_queries
            
            logger.info("📋 FINAL EVALUATION PLAN:")
            logger.info(f"  • Total queries in dataset: {len(all_queries)}")
            logger.info(f"  • Queries to use per emotion: {args.num_queries} (same for all)")
            logger.info(f"  • Emotions to evaluate: {target_emotions}")
            logger.info(f"  • Expected total evaluations: {expected_total_queries}")
            logger.info(f"  • Evaluation strategy: Same queries for ALL emotions (fair comparison)")
            
            if len(target_emotions) == 0:
                logger.error("❌ No emotions available for evaluation!")
                sys.exit(1)
        else:
            # Layerwise mode - no need for target_emotions logic
            target_emotions = [args.target_emotion]
        
        # Initialize evaluator
        start_time = time.time()
        evaluator = EmotionVectorEvaluator(
            model_name=args.model_name,
            device=args.device
        )
        
        init_time = time.time() - start_time
        logger.info(f"Model initialization completed in {init_time:.2f} seconds")
        
        # Load emotion vectors
        logger.info("Loading emotion vectors...")
        emotion_vectors = evaluator.load_emotion_vectors(args.vectors_dir)
        
        if not emotion_vectors:
            logger.error("No emotion vectors found!")
            sys.exit(1)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        evaluation_start = time.time()
        
        if args.layerwise:
            # Layerwise evaluation
            results = evaluator.evaluate_emotion_vectors_layerwise(
                emotion_vectors=emotion_vectors,
                all_queries=all_queries,
                alpha=args.alpha,
                num_queries=args.num_queries,
                target_emotion=args.target_emotion
            )
        else:
            # Standard evaluation
            results = evaluator.evaluate_emotion_vectors(
                emotion_vectors=emotion_vectors,
                all_queries=all_queries,
                alpha=args.alpha,
                num_queries_per_emotion=args.num_queries,
                target_emotions=target_emotions if args.emotion != "all" else None
            )
        
        evaluation_time = time.time() - evaluation_start
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Save results
        logger.info("Saving results...")
        if args.layerwise:
            evaluator.save_layerwise_results(results, args.output_file)
            evaluator.save_layerwise_summary(results, summary_file)
        else:
            evaluator.save_evaluation_results(results, args.output_file)
            evaluator.save_emotion_summary(results, summary_file)
        
        # Print summary statistics
        if results:
            total_time = time.time() - start_time
            
            if args.layerwise:
                avg_change = np.mean([r["confidence_change"] for r in results])
                positive_changes = len([r for r in results if r["confidence_change"] > 0])
                layers_tested = len(set([r["layer"] for r in results]))
                
                logger.info("="*80)
                logger.info("LAYERWISE EVALUATION COMPLETE!")
                logger.info("="*80)
                logger.info(f"Total time: {total_time:.2f} seconds")
                logger.info(f"Total evaluations: {len(results)}")
                logger.info(f"Layers tested: {layers_tested}")
                logger.info(f"Queries per layer: {len(results) // layers_tested}")
                logger.info(f"Average confidence change: {avg_change:+.4f}")
                logger.info(f"Positive changes: {positive_changes}/{len(results)} ({positive_changes/len(results)*100:.1f}%)")
                logger.info(f"Results saved to: {os.path.abspath(args.output_file)}")
                logger.info(f"Summary saved to: {os.path.abspath(summary_file)}")
                logger.info("="*80)
            else:
                avg_improvement = np.mean([r["confidence_improvement"] for r in results])
                positive_improvements = len([r for r in results if r["confidence_improvement"] > 0])
                
                logger.info("="*80)
                logger.info("EVALUATION COMPLETE!")
                logger.info("="*80)
                logger.info(f"Total time: {total_time:.2f} seconds")
                logger.info(f"Total evaluations: {len(results)}")
                logger.info(f"Average confidence improvement: {avg_improvement:+.4f}")
                logger.info(f"Positive improvements: {positive_improvements}/{len(results)} ({positive_improvements/len(results)*100:.1f}%)")
                logger.info(f"Results saved to: {os.path.abspath(args.output_file)}")
                logger.info(f"Summary saved to: {os.path.abspath(summary_file)}")
                logger.info("="*80)
        else:
            logger.warning("No evaluation results generated")
            
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