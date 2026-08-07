#!/usr/bin/env python3
"""
GPT-4o-mini EAS Score Calculator - Batch Processing

This script uses Azure OpenAI GPT-4o-mini to calculate emotion scores in batches
for randomly selected queries from multiple datasets:
- 50 neutral queries from full_dataset
- 50 non-neutral queries from full_dataset  
- 50 neutral queries from emotion_dataset
- 50 emotional queries from emotion_dataset

Total: 2000 queries processed in batches with comprehensive logging.

Requirements:
- openai library: pip install openai
- Valid Azure OpenAI API key
"""

import pandas as pd
import numpy as np
import json
import random
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from openai import AzureOpenAI
import warnings
warnings.filterwarnings('ignore')

# Credentials come from qwen_testing/.env (gitignored), never from source.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env_config import get_env, require_env

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class GPTEASCalculator:
    """
    Calculator for emotion scores using Azure OpenAI GPT-4o-mini API
    """
    
    def __init__(self):
        """
        Initialize the GPT EAS Calculator with Azure OpenAI configuration
        """
        self.target_emotions = ["anger", "disgust", "fear", "joy", "sadness"]

        # Azure OpenAI configuration — read from qwen_testing/.env (gitignored)
        # or from real environment variables, which take precedence.
        self.endpoint = require_env("AZURE_OPENAI_ENDPOINT")
        self.model_name = get_env("AZURE_OPENAI_MODEL", "gpt-4o-mini")
        self.deployment = get_env("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self.subscription_key = require_env("AZURE_OPENAI_API_KEY")
        self.api_version = get_env("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
        )
        
        print(f"✅ GPT-4o-mini Azure OpenAI client initialized!")
        print(f"🎯 Target emotions: {self.target_emotions}")
        print(f"🌐 Endpoint: {self.endpoint}")
        print(f"📋 Model: {self.model_name}")
    
    def create_emotion_prompt(self, query: str) -> str:
        """
        Create the emotion scoring prompt for GPT-4o-mini (same as Gemini prompt)
        
        Args:
            query: The text query to analyze
            
        Returns:
            Formatted prompt string
        """
        # Clean the query to avoid potential issues
        clean_query = query.strip().replace('"', "'").replace('\n', ' ').replace('\r', ' ')
        
        prompt = f"""Please generate the emotion scores for the following five emotions (anger, disgust, fear, joy, and sadness) based on the given sentence. Each emotion score should be a value between 0 and 100, where 0 represents no presence of the emotion, and 100 represents the maximum intensity of that emotion. Return the results in a JSON format, with the emotion names as keys and their corresponding scores as values.

You must give your response in the following JSON-string format and **DON'T** include any other text in the response.:
{{
"anger": int(0-100),
"disgust": int(0-100),
"fear": int(0-100),
"joy": int(0-100),
"sadness": int(0-100),
"surprise": int(0-100)
}}

The sentences you need to score come from a set of dialogues, and you need to score the sentiment of the **query** part.

Query: {query}

Please make sure to provide the emotion scores for the **query** part only."""
        
        return prompt
    
    def get_emotion_scores_batch(self, queries: List[str], batch_size: int = 10) -> List[Optional[Dict[str, int]]]:
        """
        Get emotion scores for a batch of queries using GPT-4o-mini
        
        Args:
            queries: List of text queries to analyze
            batch_size: Number of queries to process in parallel (simulated sequential for API limits)
            
        Returns:
            List of dictionaries with emotion scores or None for failed queries
        """
        results = []
        total_queries = len(queries)
        
        print(f"\n🔄 Processing {total_queries} queries in batches of {batch_size}...")
        
        for i in range(0, total_queries, batch_size):
            batch = queries[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_queries + batch_size - 1) // batch_size
            
            print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch)} queries)...")
            
            batch_results = []
            for j, query in enumerate(batch):
                query_num = i + j + 1
                print(f"      🔍 Query {query_num}/{total_queries}: {query[:50]}...")
                
                emotion_scores = self.get_single_emotion_scores(query)
                batch_results.append(emotion_scores)
                
                # Small delay between individual requests to avoid rate limiting
                if j < len(batch) - 1:  # Don't wait after the last query in batch
                    time.sleep(0.50)
            
            results.extend(batch_results)
            
            # Longer delay between batches
            if i + batch_size < total_queries:
                print(f"      ⏳ Waiting 2 seconds before next batch...")
                time.sleep(2)
        
        successful = sum(1 for r in results if r is not None)
        print(f"   ✅ Batch processing complete: {successful}/{total_queries} successful")
        
        return results
    
    def get_single_emotion_scores(self, query: str, max_retries: int = 3) -> Optional[Dict[str, int]]:
        """
        Get emotion scores for a single query from GPT-4o-mini API
        
        Args:
            query: Text to analyze
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with emotion scores or None if failed
        """
        if not query or pd.isna(query):
            return None
        
        prompt = self.create_emotion_prompt(query)
        
        for attempt in range(max_retries):
            try:
                # Make API call to GPT-4o-mini
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert emotion analysis assistant. Provide accurate emotion scores in the requested JSON format only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=300,
                    temperature=0.3,  # Low temperature for consistent results
                    top_p=0.950,
                    model=self.deployment
                )
                
                # Extract response content
                response_text = response.choices[0].message.content.strip()
                
                # Clean and parse JSON response
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    response_text = response_text.replace('```', '').strip()
                
                # Parse JSON
                emotion_scores = json.loads(response_text)
                
                # Validate response format
                if not isinstance(emotion_scores, dict):
                    raise ValueError("Response is not a dictionary")
                
                # Ensure all required emotions are present
                required_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
                for emotion in required_emotions:
                    if emotion not in emotion_scores:
                        raise ValueError(f"Missing emotion: {emotion}")
                    
                    # Convert to int and validate range
                    try:
                        score = int(emotion_scores[emotion])
                        if not (0 <= score <= 100):
                            raise ValueError(f"Score out of range for {emotion}: {score}")
                        emotion_scores[emotion] = score
                    except (ValueError, TypeError):
                        raise ValueError(f"Invalid score for {emotion}: {emotion_scores[emotion]}")
                
                return emotion_scores
                
            except json.JSONDecodeError as e:
                print(f"         ⚠️  JSON decode error (attempt {attempt + 1}): {e}")
                
            except Exception as e:
                print(f"         ⚠️  API error (attempt {attempt + 1}): {e}")
                
            # Wait before retry
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"         ⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        print(f"         ❌ Failed to get emotion scores after {max_retries} attempts")
        return None
    
    def calculate_eas_score(self, emotion_scores: Dict[str, int]) -> float:
        """
        Calculate EAS (Emotion Activation Score) from emotion scores
        
        EAS = sum of squared normalized emotion scores
        
        Args:
            emotion_scores: Dictionary with emotion scores (0-100)
            
        Returns:
            EAS score
        """
        if not emotion_scores:
            return 0.0
        
        # Use only the 50 target emotions for EAS calculation
        target_scores = []
        for emotion in self.target_emotions:
            if emotion in emotion_scores:
                normalized_score = emotion_scores[emotion] / 100.0  # Normalize to 0-1
                target_scores.append(normalized_score ** 2)  # Square for EAS formula
        
        if not target_scores:
            return 0.0
        
        eas_score = sum(target_scores)
        return eas_score

def load_full_dataset_queries(dataset_folder: str, target_count_per_type: int = 50) -> Tuple[List[str], List[str]]:
    """
    Load queries from full_dataset CSVs (goemotions files)
    
    Args:
        dataset_folder: Path to full_dataset folder
        target_count_per_type: Number of neutral and non-neutral queries to extract
        
    Returns:
        Tuple of (neutral_queries, non_neutral_queries)
    """
    print(f"\n📁 Loading full_dataset from: {dataset_folder}")
    
    all_neutral_queries = []
    all_non_neutral_queries = []
    
    # Expected files in full_dataset
    csv_files = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']
    
    for filename in csv_files:
        filepath = os.path.join(dataset_folder, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                print(f"   📄 Loading {filename}: {len(df)} rows")
                
                # Check for neutral column (case insensitive)
                cols_lower = [c.lower() for c in df.columns]
                if 'neutral' not in cols_lower:
                    print(f"      ⚠️  No 'neutral' column found in {filename}")
                    continue
                
                neutral_col = df.columns[cols_lower.index('neutral')]
                
                # Split by neutral value
                neutral_mask = df[neutral_col] == 1
                neutral_rows = df[neutral_mask]
                non_neutral_rows = df[~neutral_mask]
                
                # Extract text queries
                text_col = 'text' if 'text' in df.columns else df.columns[1]  # Assume second column is text
                
                neutral_texts = neutral_rows[text_col].dropna().tolist()
                non_neutral_texts = non_neutral_rows[text_col].dropna().tolist()
                
                all_neutral_queries.extend(neutral_texts)
                all_non_neutral_queries.extend(non_neutral_texts)
                
                print(f"      ✅ Added {len(neutral_texts)} neutral, {len(non_neutral_texts)} non-neutral")
                
            except Exception as e:
                print(f"      ❌ Error loading {filename}: {e}")
    
    print(f"\n📊 Full dataset totals:")
    print(f"   Neutral: {len(all_neutral_queries):,}")
    print(f"   Non-neutral: {len(all_non_neutral_queries):,}")
    
    # Sample random queries
    neutral_sample = random.sample(all_neutral_queries, min(target_count_per_type, len(all_neutral_queries)))
    non_neutral_sample = random.sample(all_non_neutral_queries, min(target_count_per_type, len(all_non_neutral_queries)))
    
    print(f"   Selected for analysis:")
    print(f"   Neutral: {len(neutral_sample)}")
    print(f"   Non-neutral: {len(non_neutral_sample)}")
    
    return neutral_sample, non_neutral_sample

def load_emotion_dataset_queries(dataset_folder: str, target_count_per_type: int = 50) -> Tuple[List[str], List[str]]:
    """
    Load queries from emotion_dataset CSVs
    
    Args:
        dataset_folder: Path to emotion_dataset folder
        target_count_per_type: Number of neutral and emotional queries to extract
        
    Returns:
        Tuple of (neutral_queries, emotional_queries)
    """
    print(f"\n📁 Loading emotion_dataset from: {dataset_folder}")
    
    neutral_file = os.path.join(dataset_folder, "2018_E_c_En_train_neutral.csv")
    emotional_file = os.path.join(dataset_folder, "2018_E_c_En_train_emotional.csv")
    
    neutral_queries = []
    emotional_queries = []
    
    # Load neutral queries
    if os.path.exists(neutral_file):
        try:
            df_neutral = pd.read_csv(neutral_file)
            print(f"   📄 Loading neutral emotions: {len(df_neutral)} rows")
            
            # Extract text (Tweet column)
            text_col = 'Tweet' if 'Tweet' in df_neutral.columns else df_neutral.columns[1]
            neutral_texts = df_neutral[text_col].dropna().tolist()
            neutral_queries = random.sample(neutral_texts, min(target_count_per_type, len(neutral_texts)))
            
            print(f"      ✅ Selected {len(neutral_queries)} neutral emotion queries")
            
        except Exception as e:
            print(f"      ❌ Error loading neutral emotions: {e}")
    
    # Load emotional queries  
    if os.path.exists(emotional_file):
        try:
            df_emotional = pd.read_csv(emotional_file)
            print(f"   📄 Loading emotional entries: {len(df_emotional)} rows")
            
            # Extract text (Tweet column)
            text_col = 'Tweet' if 'Tweet' in df_emotional.columns else df_emotional.columns[1]
            emotional_texts = df_emotional[text_col].dropna().tolist()
            emotional_queries = random.sample(emotional_texts, min(target_count_per_type, len(emotional_texts)))
            
            print(f"      ✅ Selected {len(emotional_queries)} emotional queries")
            
        except Exception as e:
            print(f"      ❌ Error loading emotional entries: {e}")
    
    print(f"\n📊 Emotion dataset totals:")
    print(f"   Neutral: {len(neutral_queries)}")
    print(f"   Emotional: {len(emotional_queries)}")
    
    return neutral_queries, emotional_queries

def analyze_all_datasets(calculator: GPTEASCalculator) -> Dict:
    """
    Analyze queries from all datasets and calculate EAS scores
    
    Args:
        calculator: GPTEASCalculator instance
        
    Returns:
        Analysis results dictionary
    """
    print(f"\n🤖 Starting comprehensive GPT-4o-mini EAS analysis...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'provider': 'Azure OpenAI',
            'model': calculator.model_name,
            'deployment': calculator.deployment,
            'endpoint': calculator.endpoint
        },
        'datasets_analyzed': {
            'full_dataset_neutral': [],
            'full_dataset_non_neutral': [], 
            'emotion_dataset_neutral': [],
            'emotion_dataset_emotional': []
        },
        'summary_stats': {}
    }
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load and analyze full_dataset
    print("\n" + "="*60)
    print("📊 PROCESSING FULL DATASET (GOEMOTIONS)")
    print("="*60)
    
    full_neutral, full_non_neutral = load_full_dataset_queries("./full_dataset", 50)
    
    if full_neutral:
        print(f"\n🔍 Analyzing {len(full_neutral)} full_dataset NEUTRAL queries...")
        neutral_scores = calculator.get_emotion_scores_batch(full_neutral, batch_size=10)
        
        for i, (query, emotion_scores) in enumerate(zip(full_neutral, neutral_scores)):
            if emotion_scores:
                eas_score = calculator.calculate_eas_score(emotion_scores)
                
                result = {
                    'query_id': i + 1,
                    'query_text': query,
                    'dataset_source': 'full_dataset',
                    'query_type': 'neutral',
                    'emotion_scores': emotion_scores,
                    'eas_score': eas_score
                }
                results['datasets_analyzed']['full_dataset_neutral'].append(result)
    
    if full_non_neutral:
        print(f"\n🔍 Analyzing {len(full_non_neutral)} full_dataset NON-NEUTRAL queries...")
        non_neutral_scores = calculator.get_emotion_scores_batch(full_non_neutral, batch_size=10)
        
        for i, (query, emotion_scores) in enumerate(zip(full_non_neutral, non_neutral_scores)):
            if emotion_scores:
                eas_score = calculator.calculate_eas_score(emotion_scores)
                
                result = {
                    'query_id': i + 1,
                    'query_text': query,
                    'dataset_source': 'full_dataset',
                    'query_type': 'non_neutral',
                    'emotion_scores': emotion_scores,
                    'eas_score': eas_score
                }
                results['datasets_analyzed']['full_dataset_non_neutral'].append(result)
    
    # Load and analyze emotion_dataset
    print("\n" + "="*60)
    print("📊 PROCESSING EMOTION DATASET (2018-E-c)")
    print("="*60)
    
    emotion_neutral, emotion_emotional = load_emotion_dataset_queries("./emotion_dataset", 50)
    
    if emotion_neutral:
        print(f"\n🔍 Analyzing {len(emotion_neutral)} emotion_dataset NEUTRAL queries...")
        emotion_neutral_scores = calculator.get_emotion_scores_batch(emotion_neutral, batch_size=10)
        
        for i, (query, emotion_scores) in enumerate(zip(emotion_neutral, emotion_neutral_scores)):
            if emotion_scores:
                eas_score = calculator.calculate_eas_score(emotion_scores)
                
                result = {
                    'query_id': i + 1,
                    'query_text': query,
                    'dataset_source': 'emotion_dataset',
                    'query_type': 'neutral',
                    'emotion_scores': emotion_scores,
                    'eas_score': eas_score
                }
                results['datasets_analyzed']['emotion_dataset_neutral'].append(result)
    
    if emotion_emotional:
        print(f"\n🔍 Analyzing {len(emotion_emotional)} emotion_dataset EMOTIONAL queries...")
        emotion_emotional_scores = calculator.get_emotion_scores_batch(emotion_emotional, batch_size=10)
        
        for i, (query, emotion_scores) in enumerate(zip(emotion_emotional, emotion_emotional_scores)):
            if emotion_scores:
                eas_score = calculator.calculate_eas_score(emotion_scores)
                
                result = {
                    'query_id': i + 1,
                    'query_text': query,
                    'dataset_source': 'emotion_dataset',
                    'query_type': 'emotional',
                    'emotion_scores': emotion_scores,
                    'eas_score': eas_score
                }
                results['datasets_analyzed']['emotion_dataset_emotional'].append(result)
    
    return results

def calculate_comprehensive_statistics(results: Dict) -> Dict:
    """
    Calculate comprehensive statistics across all datasets
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        Updated results with comprehensive statistics
    """
    print(f"\n📈 Calculating comprehensive statistics...")
    
    target_emotions = ["anger", "disgust", "fear", "joy", "sadness"]
    
    # Initialize statistics structure
    stats = {
        'full_dataset': {
            'neutral': {'count': 0, 'eas_stats': {}, 'emotion_stats': {}},
            'non_neutral': {'count': 0, 'eas_stats': {}, 'emotion_stats': {}}
        },
        'emotion_dataset': {
            'neutral': {'count': 0, 'eas_stats': {}, 'emotion_stats': {}},
            'emotional': {'count': 0, 'eas_stats': {}, 'emotion_stats': {}}
        },
        'overall_comparison': {}
    }
    
    # Calculate stats for each dataset and type
    for dataset_name in ['full_dataset', 'emotion_dataset']:
        for query_type in (['neutral', 'non_neutral'] if dataset_name == 'full_dataset' else ['neutral', 'emotional']):
            key = f"{dataset_name}_{query_type}"
            data = results['datasets_analyzed'].get(key, [])
            
            if data:
                # Extract EAS scores and emotion scores
                eas_scores = [item['eas_score'] for item in data]
                
                # EAS statistics
                stats[dataset_name][query_type]['count'] = len(eas_scores)
                stats[dataset_name][query_type]['eas_stats'] = {
                    'mean': np.mean(eas_scores),
                    'median': np.median(eas_scores),
                    'std': np.std(eas_scores),
                    'min': np.min(eas_scores),
                    'max': np.max(eas_scores)
                }
                
                # Individual emotion statistics
                emotion_stats = {}
                for emotion in target_emotions + ['surprise']:
                    emotion_values = [item['emotion_scores'].get(emotion, 0) for item in data]
                    emotion_stats[emotion] = {
                        'mean': np.mean(emotion_values),
                        'median': np.median(emotion_values),
                        'std': np.std(emotion_values),
                        'min': np.min(emotion_values),
                        'max': np.max(emotion_values)
                    }
                
                stats[dataset_name][query_type]['emotion_stats'] = emotion_stats
    
    # Overall comparison
    all_neutral_eas = []
    all_emotional_eas = []
    
    # Combine all neutral scores (from both datasets)
    for item in results['datasets_analyzed']['full_dataset_neutral'] + results['datasets_analyzed']['emotion_dataset_neutral']:
        all_neutral_eas.append(item['eas_score'])
    
    # Combine all emotional scores (non_neutral + emotional)
    for item in results['datasets_analyzed']['full_dataset_non_neutral'] + results['datasets_analyzed']['emotion_dataset_emotional']:
        all_emotional_eas.append(item['eas_score'])
    
    if all_neutral_eas and all_emotional_eas:
        stats['overall_comparison'] = {
            'neutral_combined': {
                'count': len(all_neutral_eas),
                'mean_eas': np.mean(all_neutral_eas),
                'median_eas': np.median(all_neutral_eas),
                'std_eas': np.std(all_neutral_eas)
            },
            'emotional_combined': {
                'count': len(all_emotional_eas),
                'mean_eas': np.mean(all_emotional_eas),
                'median_eas': np.median(all_emotional_eas),
                'std_eas': np.std(all_emotional_eas)
            },
            'eas_difference': np.mean(all_emotional_eas) - np.mean(all_neutral_eas)
        }
    
    results['summary_stats'] = stats
    
    # Print summary
    print(f"\n📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"=" * 50)
    
    print(f"\n📋 FULL DATASET (GoEmotions):")
    if stats['full_dataset']['neutral']['count'] > 0:
        print(f"   🔘 Neutral: {stats['full_dataset']['neutral']['count']} queries")
        print(f"      EAS Mean: {stats['full_dataset']['neutral']['eas_stats']['mean']:.4f}")
    if stats['full_dataset']['non_neutral']['count'] > 0:
        print(f"   😊 Non-neutral: {stats['full_dataset']['non_neutral']['count']} queries") 
        print(f"      EAS Mean: {stats['full_dataset']['non_neutral']['eas_stats']['mean']:.4f}")
    
    print(f"\n📋 EMOTION DATASET (2018-E-c):")
    if stats['emotion_dataset']['neutral']['count'] > 0:
        print(f"   🔘 Neutral: {stats['emotion_dataset']['neutral']['count']} queries")
        print(f"      EAS Mean: {stats['emotion_dataset']['neutral']['eas_stats']['mean']:.4f}")
    if stats['emotion_dataset']['emotional']['count'] > 0:
        print(f"   🎭 Emotional: {stats['emotion_dataset']['emotional']['count']} queries")
        print(f"      EAS Mean: {stats['emotion_dataset']['emotional']['eas_stats']['mean']:.4f}")
    
    if 'overall_comparison' in stats and stats['overall_comparison']:
        eas_diff = stats['overall_comparison']['eas_difference']
        print(f"\n📈 OVERALL COMPARISON:")
        print(f"   Combined Neutral EAS: {stats['overall_comparison']['neutral_combined']['mean_eas']:.4f}")
        print(f"   Combined Emotional EAS: {stats['overall_comparison']['emotional_combined']['mean_eas']:.4f}")
        print(f"   Difference: {eas_diff:+.4f}")
    
    return results

def save_comprehensive_results(results: Dict, output_path: str):
    """
    Save comprehensive results to JSON file with detailed structure
    
    Args:
        results: Analysis results
        output_path: Output file path (without extension)
    """
    print(f"\n💾 Saving comprehensive results...")
    
    # Add metadata
    results['metadata'] = {
        'total_queries_analyzed': sum([
            len(results['datasets_analyzed']['full_dataset_neutral']),
            len(results['datasets_analyzed']['full_dataset_non_neutral']),
            len(results['datasets_analyzed']['emotion_dataset_neutral']),
            len(results['datasets_analyzed']['emotion_dataset_emotional'])
        ]),
        'datasets_processed': ['full_dataset', 'emotion_dataset'],
        'emotions_analyzed': ["anger", "disgust", "fear", "joy", "sadness", "surprise"],
        'eas_calculation_method': 'sum_of_squared_normalized_scores',
        'random_seed_used': 42,
        'batch_processing_enabled': True,
        'api_provider': 'Azure OpenAI GPT-4o-mini'
    }
    
    # Save main results file
    json_path = f"{output_path}_comprehensive.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"✅ Comprehensive results saved to: {json_path}")
    
    # Create summary statistics file
    summary_data = {
        'analysis_timestamp': results['timestamp'],
        'model_used': results['model_info'],
        'summary_statistics': results.get('summary_stats', {}),
        'total_queries_by_type': {
            'full_dataset_neutral': len(results['datasets_analyzed']['full_dataset_neutral']),
            'full_dataset_non_neutral': len(results['datasets_analyzed']['full_dataset_non_neutral']),
            'emotion_dataset_neutral': len(results['datasets_analyzed']['emotion_dataset_neutral']),
            'emotion_dataset_emotional': len(results['datasets_analyzed']['emotion_dataset_emotional'])
        }
    }
    
    summary_path = f"{output_path}_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"✅ Summary statistics saved to: {summary_path}")
    
    return json_path, summary_path

def main():
    """Main execution function"""
    print("🚀 Starting GPT-4o-mini Comprehensive EAS Score Analysis")
    print("=" * 70)
    print("📊 Target: 2000 queries total (50 from each category)")
    print("🔄 Batch processing enabled for efficiency")
    
    try:
        # Initialize calculator
        calculator = GPTEASCalculator()
        
        # Analyze all datasets
        results = analyze_all_datasets(calculator)
        
        # Calculate comprehensive statistics
        results = calculate_comprehensive_statistics(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"gpt4o_comprehensive_eas_analysis_{timestamp}"
        
        json_path, summary_path = save_comprehensive_results(results, output_path)
        
        print(f"\n🎉 Comprehensive analysis complete!")
        print(f"📄 Results saved to:")
        print(f"   📊 Main results: {json_path}")
        print(f"   📋 Summary: {summary_path}")
        
        print(f"\n🎯 Analysis includes:")
        print(f"   • Full dataset (GoEmotions): neutral + non-neutral queries")
        print(f"   • Emotion dataset (2018-E-c): neutral + emotional queries") 
        print(f"   • Comprehensive cross-dataset comparisons")
        print(f"   • Individual emotion breakdowns")
        print(f"   • EAS score distributions and statistics")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()