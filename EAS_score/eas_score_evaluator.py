#!/usr/bin/env python3
"""
EAS Score Evaluator for Emotion Vectors

This script evaluates the Emotion Appropriateness Score (EAS) by:
1. Loading pre-extracted emotion vectors from JSON files
2. Generating responses with and without emotion vector injection (steering)
3. Using GPT-4o-mini API to score absolute emotion strength across 6 basic emotions
4. Calculating EAS scores using the research formula: EAS = sum((score_emotion/100)^2) for all emotions

EAS Methodology (Based on Research Paper):
- Uses EmotionQuery+ dataset (not emotion-specific queries)
- Scores each response on 6 basic emotions: anger, disgust, fear, joy, sadness, surprise
- Each emotion scored 0-100 (absolute intensity, not appropriateness)
- EAS = sum((score_emotion/100)^2) across all 6 emotions
- EAS range: 0.0 to 6.0 (theoretical maximum if all emotions = 100)
- EAS > 0.5: Clear emotional tendency
- EAS > 1.0: Strong emotion(s) present

Outputs:
- CSV file with all individual results, emotion scores, and responses
- JSON summary file with statistical analysis

Usage:
    python eas_score_evaluator.py --alpha 2.0 --num_queries 100 --api_key <your-key>
    python eas_score_evaluator.py --alpha 1.5 --emotion joy --num_queries 50 --api_key <your-key>

Author: EAS Evaluation System (Research Paper Formula)
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

# OpenAI imports
try:
    from openai import AzureOpenAI
except ImportError:
    print("Error: openai library not found. Please install with: pip install openai")
    sys.exit(1)

# Transformers imports
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM
    )
except ImportError:
    print("Error: transformers library not found. Please install with: pip install transformers torch")
    sys.exit(1)

# Credentials come from qwen_testing/.env (gitignored), never from source.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env_config import get_env, require_env

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eas_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EASScoreEvaluator:
    """
    EAS Score Evaluator for testing emotion vector effectiveness using GPT-4o-mini
    
    Evaluates emotion vectors by:
    1. Loading pre-extracted emotion vectors
    2. Generating responses with/without emotion vectors
    3. Using GPT-4o-mini to score emotion appropriateness
    4. Calculating EAS scores following research methodology
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 api_key: str = None,
                 device: str = "cuda",
                 torch_dtype: torch.dtype = torch.float32):
        """
        Initialize the EAS Score Evaluator
        
        Args:
            model_name: HuggingFace model identifier for generation
            api_key: Azure OpenAI API key
            device: Device to run models on
            torch_dtype: PyTorch data type for model weights
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Initialize emotion vector injection variables
        self.hooks = []
        self.current_emotion_vectors = None
        self.current_alpha = 0.0
        self.layer_idx = 0
        
        logger.info(f"Initializing EAS Score Evaluator")
        logger.info(f"Generation Model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Data Type: {torch_dtype}")
        
        # Load generation model
        self._load_generation_model()
        
        # Initialize GPT-4o-mini client
        self._initialize_gpt_client(api_key)
        
        # Define emotion categories
        self.emotions = ["base"]
        
        logger.info(f"EAS Evaluator initialized successfully")
        logger.info(f"Target emotions: {self.emotions}")
    
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
    
    def _initialize_gpt_client(self, api_key: str):
        """Initialize GPT-4o-mini client for EAS scoring"""
        
        logger.info("Initializing GPT-4o-mini client...")
        try:
            # fall back to the environment when --api_key was not passed
            api_key = api_key or require_env("AZURE_OPENAI_API_KEY")
            
            # Azure OpenAI configuration — from qwen_testing/.env (gitignored)
            # or from real environment variables, which take precedence.
            endpoint = require_env("AZURE_OPENAI_ENDPOINT")
            model_name = get_env("AZURE_OPENAI_MODEL", "gpt-4o-mini")
            deployment = get_env("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
            api_version = get_env("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            
            self.gpt_client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
            
            self.gpt_deployment = deployment
            
            logger.info(f"GPT-4o-mini client initialized successfully!")
            logger.info(f"Endpoint: {endpoint}")
            logger.info(f"Model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4o-mini client: {e}")
            raise
    
    def load_emotion_vectors(self, vectors_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load emotion vectors from JSON files"""
        
        logger.info(f"Loading emotion vectors from: {vectors_dir}")
        
        emotion_vectors = {}
        
        for emotion in self.emotions:
            vector_file = os.path.join(vectors_dir, f"Qwen25_1B5_{emotion}_vectors.json")
            
            if os.path.exists(vector_file):
                try:
                    with open(vector_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract layer vectors
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
                        logger.info(f"Loaded {emotion}: {len(layer_vectors)} layers (avg norm: {avg_norm:.3f})")
                    
                except Exception as e:
                    logger.error(f"Failed to load {emotion} vectors: {e}")
                    continue
            else:
                logger.warning(f"Vector file not found: {vector_file}")
        
        logger.info(f"Loaded emotion vectors for {len(emotion_vectors)} emotions")
        return emotion_vectors
    
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
        """Create a forward hook for emotion vector injection"""
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
                            
                            # Normalize the emotion vector to unit length (L2 normalization)
                            emotion_vector_norm = torch.nn.functional.normalize(emotion_vector, p=2, dim=0)
                            
                            # Apply emotion vector to the last token's hidden state
                            if isinstance(output_tensor, tuple):
                                hidden_states = output_tensor[0]
                            else:
                                hidden_states = output_tensor
                            
                            # Inject normalized emotion vector with alpha multiplier to the last token
                            if len(hidden_states.shape) == 3:
                                hidden_states[:, -1:, :] = (
                                    hidden_states[:, -1:, :] + 
                                    self.current_alpha * emotion_vector_norm.unsqueeze(0).unsqueeze(0).to(hidden_states.device)
                                )
                            
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
    
    def _remove_emotion_hooks(self):
        """Remove all registered forward hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_idx = 0

    def generate_with_emotion_vector(self, 
                                   prompt: str, 
                                   emotion_vectors: Optional[Dict[int, torch.Tensor]] = None,
                                   alpha: float = 1.0,
                                   max_new_tokens: int = 100) -> str:
        """Generate text with emotion vector injection via model hooks"""
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
                self.current_emotion_vectors = emotion_vectors
                self.current_alpha = alpha
                self._register_emotion_hooks()
            else:
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
    
    def calculate_emotion_scores(self, query: str, response: str) -> Dict[str, int]:
        """
        Calculate absolute emotion scores for all 6 basic emotions using GPT-4o-mini
        
        Args:
            query: The input query/prompt
            response: The generated response to evaluate
            
        Returns:
            Dictionary with emotion scores (0-100 for each emotion)
        """
        try:
            # Create the evaluation prompt following research methodology from the paper
            evaluation_prompt = f"""Please generate the emotion scores for the following six emotions (anger, disgust, fear, joy, sadness, and surprise) based on the given sentence. Each emotion score should be a value between 0 and 100, where 0 represents no presence of the emotion, and 100 represents the maximum intensity of that emotion.

Return the results in a JSON format, with the emotion names as keys and their corresponding scores as values.

You must give your response in the following JSON-string format and **DON'T** include any other text in the response.:
{{
"anger": int(0-100),
"disgust": int(0-100),
"fear": int(0-100),
"joy": int(0-100),
"sadness": int(0-100),
"surprise": int(0-100)
}}

The sentences you need to score come from a set of dialogues, and you need to score the sentiment of the **answer** part.

Question: {query}
Answer: {response}

Please make sure to provide the emotion scores for the **answer** part only."""

            # Call GPT-4o-mini API with rate limiting
            time.sleep(0.1)  # Small delay to avoid rate limiting
            response_obj = self.gpt_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert emotion evaluator. You analyze text and provide precise emotion scores for 6 basic emotions. Always respond with only a valid JSON object containing integer scores from 0-100."
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ],
                max_tokens=100,
                temperature=0.1,  # Low temperature for consistent scoring
                top_p=1.0,
                model=self.gpt_deployment
            )
            
            # Extract and parse JSON response
            score_text = response_obj.choices[0].message.content.strip()
            
            try:
                import json
                emotion_scores = json.loads(score_text)
                
                # Validate and clamp scores
                valid_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
                cleaned_scores = {}
                
                for emotion in valid_emotions:
                    if emotion in emotion_scores:
                        score = int(emotion_scores[emotion])
                        cleaned_scores[emotion] = max(0, min(100, score))
                    else:
                        logger.warning(f"Missing emotion score for {emotion}, defaulting to 0")
                        cleaned_scores[emotion] = 0
                
                return cleaned_scores
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse emotion scores JSON: {score_text}, Error: {e}")
                # Return default scores
                return {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0, "surprise": 0}
                
        except Exception as e:
            logger.error(f"Error calculating emotion scores: {e}")
            # Return default scores
            return {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0, "surprise": 0}

    def calculate_eas_score(self, emotion_scores: Dict[str, int]) -> float:
        """
        Calculate EAS (Emotion Appropriateness Score) using the research formula:
        EAS = sum((score_emotion/100)^2) for all emotions
        
        Args:
            emotion_scores: Dictionary with emotion scores (0-100 for each emotion)
            
        Returns:
            EAS score (0.0 to 6.0 theoretical maximum)
        """
        eas = 0.0
        for emotion, score in emotion_scores.items():
            normalized_score = score / 100.0  # Convert 0-100 to 0.0-1.0
            eas += normalized_score ** 2
        
        return eas
    
    def evaluate_eas_scores(self, 
                          emotion_vectors: Dict[str, Dict[int, torch.Tensor]],
                          all_queries: List[str],
                          alpha: float,
                          num_queries: int,
                          target_emotions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate EAS scores using EmotionQuery+ dataset (not emotion-specific queries)
        
        Args:
            emotion_vectors: Loaded emotion vectors for different emotions
            all_queries: List of queries from EmotionQuery+ (general queries, not emotion-labeled)
            alpha: Multiplier for emotion vector strength
            num_queries: Number of queries to test per emotion vector
            target_emotions: Specific emotion vectors to evaluate (None for all)
            
        Returns:
            List of evaluation results with 6-emotion scores and EAS calculations
        """
        logger.info("Starting EAS score evaluation...")
        logger.info(f"Alpha multiplier: {alpha}")
        logger.info(f"Total queries available: {len(all_queries)}")
        logger.info(f"Queries to evaluate: {num_queries}")
        
        # Sample evaluation queries
        import random
        random.seed(42)  # Fixed seed for reproducibility
        
        if len(all_queries) >= num_queries:
            evaluation_queries = random.sample(all_queries, num_queries)
            logger.info(f"✅ Randomly sampled {num_queries} queries from {len(all_queries)} available")
        else:
            evaluation_queries = random.choices(all_queries, k=num_queries)
            logger.info(f"⚠️  Using sampling with replacement to get {num_queries} queries")
        
        # Determine which emotions to evaluate
        emotions_to_evaluate = target_emotions if target_emotions is not None else self.emotions
        logger.info(f"🎯 Emotions to evaluate: {emotions_to_evaluate}")
        
        all_results = []
        
        for emotion in emotions_to_evaluate:
            if emotion not in emotion_vectors:
                logger.warning(f"No vectors found for emotion: {emotion}")
                continue
            
            logger.info(f"Evaluating emotion: {emotion.upper()}")
            emotion_vector_layers = emotion_vectors[emotion]
            
            for query_idx, query in enumerate(tqdm(evaluation_queries, desc=f"EAS-{emotion}")):
                try:
                    # Create neutral system message
                    neutral_system = (
                        "You are an AI assistant. Respond to the user's query in a helpful manner."
                    )
                    
                    # Format prompt
                    prompt = self._format_qwen_prompt(neutral_system, query)
                    
                    # Generate baseline response (without emotion vector)
                    baseline_response = self.generate_with_emotion_vector(
                        prompt=prompt,
                        emotion_vectors=None,
                        alpha=0.0
                    )
                    
                    # Generate enhanced response (with emotion vector injection)
                    enhanced_response = self.generate_with_emotion_vector(
                        prompt=prompt,
                        emotion_vectors=emotion_vector_layers,
                        alpha=alpha
                    )
                    
                    # Calculate 6-emotion scores for both responses using GPT-4o-mini
                    logger.debug(f"Calculating emotion scores for {emotion} query {query_idx+1}")
                    baseline_emotion_scores = self.calculate_emotion_scores(query, baseline_response)
                    enhanced_emotion_scores = self.calculate_emotion_scores(query, enhanced_response)
                    
                    # Calculate EAS scores using research formula: EAS = sum((score/100)^2)
                    baseline_eas = self.calculate_eas_score(baseline_emotion_scores)
                    enhanced_eas = self.calculate_eas_score(enhanced_emotion_scores)
                    
                    # Calculate EAS difference (improvement)
                    eas_difference = enhanced_eas - baseline_eas
                    
                    # Store results
                    result = {
                        "emotion_vector": emotion,
                        "query_idx": query_idx,
                        "query": query,
                        "alpha": alpha,
                        "baseline_response": baseline_response,
                        "enhanced_response": enhanced_response,
                        
                        # Baseline emotion scores (0-100)
                        "baseline_anger": baseline_emotion_scores["anger"],
                        "baseline_disgust": baseline_emotion_scores["disgust"],
                        "baseline_fear": baseline_emotion_scores["fear"],
                        "baseline_joy": baseline_emotion_scores["joy"],
                        "baseline_sadness": baseline_emotion_scores["sadness"],
                        "baseline_surprise": baseline_emotion_scores["surprise"],
                        "baseline_eas": baseline_eas,
                        
                        # Enhanced emotion scores (0-100)
                        "enhanced_anger": enhanced_emotion_scores["anger"],
                        "enhanced_disgust": enhanced_emotion_scores["disgust"],
                        "enhanced_fear": enhanced_emotion_scores["fear"],
                        "enhanced_joy": enhanced_emotion_scores["joy"],
                        "enhanced_sadness": enhanced_emotion_scores["sadness"],
                        "enhanced_surprise": enhanced_emotion_scores["surprise"],
                        "enhanced_eas": enhanced_eas,
                        
                        # EAS difference
                        "eas_difference": eas_difference,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    all_results.append(result)
                    
                    logger.debug(f"Query {query_idx+1} - {emotion}: Baseline EAS {baseline_eas:.3f} → Enhanced EAS {enhanced_eas:.3f} (Δ: {eas_difference:+.3f})")
                    
                except Exception as e:
                    logger.error(f"Error evaluating query {query_idx+1}: {e}")
                    continue
        
        return all_results
    
    def save_eas_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save EAS evaluation results to CSV file and JSON summary"""
        if not results:
            logger.warning("No results to save")
            return
        
        logger.info(f"Saving {len(results)} EAS results to: {output_file}")
        
        # Define CSV columns
        fieldnames = [
            "emotion_vector", "query_idx", "alpha",
            "baseline_anger", "baseline_disgust", "baseline_fear", "baseline_joy", "baseline_sadness", "baseline_surprise", "baseline_eas",
            "enhanced_anger", "enhanced_disgust", "enhanced_fear", "enhanced_joy", "enhanced_sadness", "enhanced_surprise", "enhanced_eas",
            "eas_difference",
            "query", "baseline_response", "enhanced_response", "timestamp"
        ]
        
        try:
            # Save CSV file
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            file_size = os.path.getsize(output_file) / 1024  # Size in KB
            logger.info(f"EAS results CSV saved successfully ({file_size:.2f} KB)")
            
            # Create JSON summary
            self._save_json_summary(results, output_file)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def _save_json_summary(self, results: List[Dict[str, Any]], csv_output_file: str):
        """Save JSON summary of EAS evaluation results"""
        try:
            # Calculate summary statistics
            baseline_eas_scores = [r["baseline_eas"] for r in results]
            enhanced_eas_scores = [r["enhanced_eas"] for r in results]
            eas_differences = [r["eas_difference"] for r in results]
            
            # Overall statistics
            overall_stats = {
                "baseline_eas": {
                    "mean": float(np.mean(baseline_eas_scores)),
                    "std": float(np.std(baseline_eas_scores)),
                    "min": float(np.min(baseline_eas_scores)),
                    "max": float(np.max(baseline_eas_scores))
                },
                "enhanced_eas": {
                    "mean": float(np.mean(enhanced_eas_scores)),
                    "std": float(np.std(enhanced_eas_scores)),
                    "min": float(np.min(enhanced_eas_scores)),
                    "max": float(np.max(enhanced_eas_scores))
                },
                "eas_difference": {
                    "mean": float(np.mean(eas_differences)),
                    "std": float(np.std(eas_differences)),
                    "min": float(np.min(eas_differences)),
                    "max": float(np.max(eas_differences)),
                    "positive_count": int(np.sum([1 for diff in eas_differences if diff > 0])),
                    "negative_count": int(np.sum([1 for diff in eas_differences if diff < 0])),
                    "zero_count": int(np.sum([1 for diff in eas_differences if abs(diff) < 0.001]))
                }
            }
            
            # Per-emotion vector statistics
            emotion_stats = {}
            for emotion in self.emotions:
                emotion_results = [r for r in results if r["emotion_vector"] == emotion]
                if emotion_results:
                    emotion_baseline_eas = [r["baseline_eas"] for r in emotion_results]
                    emotion_enhanced_eas = [r["enhanced_eas"] for r in emotion_results]
                    emotion_differences = [r["eas_difference"] for r in emotion_results]
                    
                    emotion_stats[emotion] = {
                        "count": len(emotion_results),
                        "baseline_eas_mean": float(np.mean(emotion_baseline_eas)),
                        "enhanced_eas_mean": float(np.mean(emotion_enhanced_eas)),
                        "eas_difference_mean": float(np.mean(emotion_differences)),
                        "eas_difference_std": float(np.std(emotion_differences)),
                        "positive_improvements": int(np.sum([1 for diff in emotion_differences if diff > 0])),
                        "negative_impacts": int(np.sum([1 for diff in emotion_differences if diff < 0]))
                    }
            
            # Create summary object
            summary = {
                "evaluation_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_evaluations": len(results),
                    "alpha": results[0]["alpha"] if results else None,
                    "emotions_evaluated": list(set([r["emotion"] for r in results])),
                    "model_name": self.model_name
                },
                "overall_statistics": overall_stats,
                "per_emotion_statistics": emotion_stats,
                "methodology": {
                    "eas_formula": "EAS = sum((score_emotion/100)^2) for all 6 emotions",
                    "emotion_scoring": "6 basic emotions (anger, disgust, fear, joy, sadness, surprise) scored 0-100",
                    "eas_range": "0.0 to 6.0 (theoretical maximum if all emotions = 100)",
                    "eas_interpretation": {
                        "0.0-0.5": "Minimal emotional content",
                        "0.5-1.0": "Clear emotional tendency",
                        ">1.0": "Strong emotion(s) present"
                    },
                    "eas_difference": "Enhanced EAS - Baseline EAS",
                    "positive_difference_meaning": "Emotion vector increased emotional intensity",
                    "negative_difference_meaning": "Emotion vector decreased emotional intensity",
                    "evaluation_model": "GPT-4o-mini via Azure OpenAI",
                    "dataset": "EmotionQuery+ (not emotion-specific queries)"
                }
            }
            
            # Save JSON summary
            json_file = csv_output_file.replace('.csv', '_summary.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON summary saved to: {json_file}")
            
        except Exception as e:
            logger.error(f"Error saving JSON summary: {e}")
            raise

def load_emotion_queries(emotion_queries_file: str) -> List[str]:
    """Load emotion queries from JSON file"""
    logger.info(f"Loading emotion queries from: {emotion_queries_file}")
    
    try:
        with open(emotion_queries_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            logger.info(f"✅ Loaded {len(data)} queries from flat list format")
            return data
        elif isinstance(data, dict):
            # If organized by emotion, flatten to single list
            all_queries = []
            for emotion, queries in data.items():
                if isinstance(queries, list):
                    all_queries.extend(queries)
            logger.info(f"✅ Total queries collected: {len(all_queries)}")
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

def main():
    """Main function to run EAS score evaluation"""
    
    parser = argparse.ArgumentParser(
        description="Evaluate emotion vectors using EAS scores with GPT-4o-mini",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--alpha", 
        type=float, 
        required=True,
        help="Multiplier for emotion vector strength (e.g., 1.0, 2.0, 3.0)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Azure OpenAI API key. Defaults to AZURE_OPENAI_API_KEY from "
             "qwen_testing/.env — prefer that over passing it on the command "
             "line, which leaks the key into shell history."
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
        default=50,
        help="Number of queries to evaluate"
    )
    parser.add_argument(
        "--emotion", 
        type=str, 
        choices=["joy", "anger", "disgust", "fear", "sadness", "all"],
        default="all",
        help="Specific emotion to evaluate (default: all)"
    )
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
        emotion_suffix = f"_{args.emotion}" if args.emotion != "all" else ""
        args.output_file = f"eas_evaluation_alpha{args.alpha}{emotion_suffix}_{timestamp}.csv"
    
    logger.info("="*80)
    logger.info("EAS SCORE EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Alpha multiplier: {args.alpha}")
    logger.info(f"Number of queries: {args.num_queries}")
    logger.info(f"Target emotion: {args.emotion}")
    logger.info(f"EAS Scoring: GPT-4o-mini API")
    logger.info(f"Vectors directory: {args.vectors_dir}")
    logger.info(f"Output file: {args.output_file}")
    logger.info("="*80)
    
    # Check device availability
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    try:
        # Load emotion queries
        all_queries = load_emotion_queries(args.emotion_queries_file)
        
        logger.info(f"📊 Total queries loaded: {len(all_queries)}")
        logger.info(f"📋 Queries to evaluate: {args.num_queries}")
        
        # Determine target emotions
        target_emotions = None if args.emotion == "all" else [args.emotion]
        
        # Initialize evaluator
        start_time = time.time()
        evaluator = EASScoreEvaluator(
            model_name=args.model_name,
            api_key=args.api_key,
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
        
        # Run EAS evaluation
        logger.info("Starting EAS evaluation...")
        evaluation_start = time.time()
        
        results = evaluator.evaluate_eas_scores(
            emotion_vectors=emotion_vectors,
            all_queries=all_queries,
            alpha=args.alpha,
            num_queries=args.num_queries,
            target_emotions=target_emotions
        )
        
        evaluation_time = time.time() - evaluation_start
        logger.info(f"EAS evaluation completed in {evaluation_time:.2f} seconds")
        
        # Save results
        logger.info("Saving EAS results...")
        evaluator.save_eas_results(results, args.output_file)
        
        # Print summary statistics
        if results:
            total_time = time.time() - start_time
            avg_baseline_eas = np.mean([r["baseline_eas"] for r in results])
            avg_enhanced_eas = np.mean([r["enhanced_eas"] for r in results])
            avg_eas_difference = np.mean([r["eas_difference"] for r in results])
            positive_improvements = len([r for r in results if r["eas_difference"] > 0])
            negative_impacts = len([r for r in results if r["eas_difference"] < 0])
            
            logger.info("="*80)
            logger.info("EAS EVALUATION COMPLETE!")
            logger.info("="*80)
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info(f"Total evaluations: {len(results)}")
            logger.info(f"Average baseline EAS: {avg_baseline_eas:.4f}")
            logger.info(f"Average enhanced EAS: {avg_enhanced_eas:.4f}")
            logger.info(f"Average EAS difference: {avg_eas_difference:+.4f}")
            logger.info(f"Positive improvements: {positive_improvements}/{len(results)} ({positive_improvements/len(results)*100:.1f}%)")
            logger.info(f"Negative impacts: {negative_impacts}/{len(results)} ({negative_impacts/len(results)*100:.1f}%)")
            logger.info(f"Results saved to: {os.path.abspath(args.output_file)}")
            logger.info(f"Summary saved to: {os.path.abspath(args.output_file.replace('.csv', '_summary.json'))}")
            logger.info("="*80)
        else:
            logger.warning("No EAS evaluation results generated")
            
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