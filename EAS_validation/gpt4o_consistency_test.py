#!/usr/bin/env python3
"""
GPT-4o-mini Consistency Test - EAS Score Variance Analysis

This script tests the consistency of GPT-4o-mini by running the same query 20 times
and calculating the average EAS score and standard deviation to measure variance.

Requirements:
- openai library: pip install openai
- Valid Azure OpenAI API key
"""

import pandas as pd
import numpy as np
import json
import time
import random
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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

class GPTConsistencyTester:
    """
    Test GPT-4o-mini consistency by running the same query multiple times
    """
    
    def __init__(self):
        """
        Initialize the GPT Consistency Tester with Azure OpenAI configuration
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
        print(f"🎯 Target emotions for EAS: {self.target_emotions}")
        print(f"🌐 Endpoint: {self.endpoint}")
        print(f"📋 Model: {self.model_name}")
    
    def create_emotion_prompt(self, query: str) -> str:
        """
        Create the emotion scoring prompt for GPT-4o-mini (exact same as gpt4o_eas_calculator.py)
        
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

Query: {clean_query}

Please make sure to provide the emotion scores for the **query** part only."""
        
        return prompt
    
    def get_single_emotion_scores(self, query: str, attempt_number: int = 1, max_retries: int = 3) -> Optional[Dict[str, int]]:
        """
        Get emotion scores for a single query from GPT-4o-mini API
        
        Args:
            query: Text to analyze
            attempt_number: Current attempt number for logging
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with emotion scores or None if failed
        """
        if not query or pd.isna(query):
            return None
        
        prompt = self.create_emotion_prompt(query)
        
        for retry in range(max_retries):
            try:
                print(f"      🔄 Attempt {attempt_number}: API call (retry {retry + 1}/{max_retries})")
                
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
                
                print(f"      ✅ Attempt {attempt_number}: Success")
                return emotion_scores
                
            except json.JSONDecodeError as e:
                print(f"         ⚠️  JSON decode error (retry {retry + 1}): {e}")
                
            except Exception as e:
                print(f"         ⚠️  API error (retry {retry + 1}): {e}")
                
            # Wait before retry
            if retry < max_retries - 1:
                wait_time = 2 ** retry  # Exponential backoff
                print(f"         ⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        print(f"         ❌ Attempt {attempt_number}: Failed after {max_retries} retries")
        return None
    
    def calculate_eas_score(self, emotion_scores: Dict[str, int]) -> float:
        """
        Calculate EAS (Emotion Activation Score) from emotion scores
        
        EAS = sum of squared normalized emotion scores (same formula as gpt4o_eas_calculator.py)
        
        Args:
            emotion_scores: Dictionary with emotion scores (0-100)
            
        Returns:
            EAS score
        """
        if not emotion_scores:
            return 0.0
        
        # Use only the 5 target emotions for EAS calculation
        target_scores = []
        for emotion in self.target_emotions:
            if emotion in emotion_scores:
                normalized_score = emotion_scores[emotion] / 100.0  # Normalize to 0-1
                target_scores.append(normalized_score ** 2)  # Square for EAS formula
        
        if not target_scores:
            return 0.0
        
        eas_score = sum(target_scores)
        return eas_score
    
    def run_single_query_consistency_test(self, test_query: str, num_iterations: int = 20) -> Dict:
        """
        Run consistency test by analyzing the same query multiple times
        
        Args:
            test_query: Query text to analyze repeatedly
            num_iterations: Number of times to run the analysis (default: 20)
            
        Returns:
            Dictionary with test results and statistics
        """
        print(f"\n🧪 Starting GPT-4o-mini consistency test")
        print(f"📝 Test Query: {test_query}")
        print(f"🔄 Iterations: {num_iterations}")
        print(f"=" * 70)
        
        results = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'model': self.model_name,
                'deployment': self.deployment,
                'endpoint': self.endpoint,
                'query': test_query,
                'iterations': num_iterations,
                'temperature': 0.3,
                'top_p': 0.950
            },
            'individual_results': [],
            'statistics': {},
            'consistency_analysis': {}
        }
        
        successful_attempts = 0
        eas_scores = []
        emotion_score_lists = {emotion: [] for emotion in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]}
        
        # Run multiple iterations
        for i in range(1, num_iterations + 1):
            print(f"\n🔍 Running iteration {i}/{num_iterations}")
            
            # Get emotion scores for this iteration
            emotion_scores = self.get_single_emotion_scores(test_query, attempt_number=i)
            
            if emotion_scores:
                successful_attempts += 1
                
                # Calculate EAS score
                eas_score = self.calculate_eas_score(emotion_scores)
                eas_scores.append(eas_score)
                
                # Store individual emotion scores
                for emotion in emotion_score_lists.keys():
                    if emotion in emotion_scores:
                        emotion_score_lists[emotion].append(emotion_scores[emotion])
                
                # Store iteration result
                iteration_result = {
                    'iteration': i,
                    'success': True,
                    'emotion_scores': emotion_scores.copy(),
                    'eas_score': eas_score,
                    'timestamp': datetime.now().isoformat()
                }
                
                results['individual_results'].append(iteration_result)
                
                print(f"      📊 EAS Score: {eas_score:.6f}")
                print(f"      😊 Joy: {emotion_scores.get('joy', 0)}, 😢 Sadness: {emotion_scores.get('sadness', 0)}")
                print(f"      😠 Anger: {emotion_scores.get('anger', 0)}, 😨 Fear: {emotion_scores.get('fear', 0)}")
                
            else:
                # Store failed iteration
                iteration_result = {
                    'iteration': i,
                    'success': False,
                    'emotion_scores': None,
                    'eas_score': None,
                    'timestamp': datetime.now().isoformat()
                }
                results['individual_results'].append(iteration_result)
                print(f"      ❌ Failed to get emotion scores")
            
            # Wait between iterations to avoid rate limiting
            if i < num_iterations:
                print(f"      ⏳ Waiting 1 second before next iteration...")
                time.sleep(1)
        
        print(f"\n📈 Calculating statistics...")
        
        # Calculate comprehensive statistics
        if eas_scores:
            results['statistics'] = {
                'success_rate': successful_attempts / num_iterations,
                'successful_iterations': successful_attempts,
                'failed_iterations': num_iterations - successful_attempts,
                'eas_statistics': {
                    'mean': float(np.mean(eas_scores)),
                    'median': float(np.median(eas_scores)),
                    'std_deviation': float(np.std(eas_scores)),
                    'variance': float(np.var(eas_scores)),
                    'min': float(np.min(eas_scores)),
                    'max': float(np.max(eas_scores)),
                    'range': float(np.max(eas_scores) - np.min(eas_scores)),
                    'coefficient_of_variation': float(np.std(eas_scores) / np.mean(eas_scores)) if np.mean(eas_scores) > 0 else 0
                },
                'individual_emotion_statistics': {}
            }
            
            # Calculate statistics for each individual emotion
            for emotion, scores in emotion_score_lists.items():
                if scores:  # Only calculate if we have scores for this emotion
                    results['statistics']['individual_emotion_statistics'][emotion] = {
                        'mean': float(np.mean(scores)),
                        'median': float(np.median(scores)),
                        'std_deviation': float(np.std(scores)),
                        'variance': float(np.var(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores)),
                        'range': float(np.max(scores) - np.min(scores)),
                        'coefficient_of_variation': float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
                    }
            
            # Consistency analysis
            eas_cv = results['statistics']['eas_statistics']['coefficient_of_variation']
            eas_std = results['statistics']['eas_statistics']['std_deviation']
            
            if eas_cv < 0.1:
                consistency_level = "Very High"
            elif eas_cv < 0.2:
                consistency_level = "High"
            elif eas_cv < 0.3:
                consistency_level = "Moderate"
            else:
                consistency_level = "Low"
            
            results['consistency_analysis'] = {
                'consistency_level': consistency_level,
                'coefficient_of_variation': eas_cv,
                'interpretation': {
                    'cv_meaning': "Coefficient of Variation (CV) = std_dev / mean. Lower CV indicates higher consistency.",
                    'cv_thresholds': {
                        'very_high': "CV < 0.1",
                        'high': "0.1 <= CV < 0.2", 
                        'moderate': "0.2 <= CV < 0.3",
                        'low': "CV >= 0.3"
                    }
                },
                'recommendations': self._generate_consistency_recommendations(eas_cv, eas_std, successful_attempts, num_iterations)
            }
        
        return results
    
    def _generate_consistency_recommendations(self, cv: float, std_dev: float, success_count: int, total_iterations: int) -> List[str]:
        """Generate recommendations based on consistency analysis"""
        recommendations = []
        
        success_rate = success_count / total_iterations
        
        if success_rate < 0.9:
            recommendations.append(f"Low success rate ({success_rate:.1%}). Consider improving error handling or API reliability.")
        
        if cv < 0.1:
            recommendations.append("Excellent consistency! GPT-4o-mini shows very stable emotion scoring for this type of query.")
        elif cv < 0.2:
            recommendations.append("Good consistency. GPT-4o-mini provides reasonably stable emotion scores.")
        elif cv < 0.3:
            recommendations.append("Moderate consistency. Consider averaging multiple runs for more stable results.")
        else:
            recommendations.append("Low consistency detected. Results may vary significantly between runs.")
        
        if std_dev > 0.5:
            recommendations.append("High standard deviation in EAS scores. Consider using ensemble methods or multiple samples.")
        
        return recommendations

def load_random_queries_from_csv(csv_path: str, num_queries: int = 20) -> List[str]:
    """
    Load random queries from the emotion dataset CSV file
    
    Args:
        csv_path: Path to the CSV file
        num_queries: Number of random queries to select
        
    Returns:
        List of selected query texts
    """
    print(f"\n📁 Loading random queries from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   📄 CSV loaded: {len(df)} total rows")
        
        # Extract Tweet column (assuming it contains the text)
        if 'Tweet' in df.columns:
            all_tweets = df['Tweet'].dropna().tolist()
            print(f"   📝 Found {len(all_tweets)} valid tweets")
            
            # Select random queries
            if len(all_tweets) >= num_queries:
                selected_queries = random.sample(all_tweets, num_queries)
                print(f"   ✅ Selected {len(selected_queries)} random queries")
                return selected_queries
            else:
                print(f"   ⚠️  Only {len(all_tweets)} queries available, using all")
                return all_tweets
        else:
            print(f"   ❌ 'Tweet' column not found in CSV. Available columns: {df.columns.tolist()}")
            return []
            
    except Exception as e:
        print(f"   ❌ Error loading CSV: {e}")
        return []

def run_multi_query_consistency_test(tester: 'GPTConsistencyTester', queries: List[str], iterations_per_query: int = 10) -> Dict:
    """
    Run consistency test on multiple queries
    
    Args:
        tester: GPTConsistencyTester instance
        queries: List of queries to test
        iterations_per_query: Number of times to test each query
        
    Returns:
        Comprehensive results dictionary
    """
    print(f"\n🧪 Starting Multi-Query Consistency Test")
    print(f"📊 Testing {len(queries)} queries with {iterations_per_query} iterations each")
    print(f"🔢 Total API calls: {len(queries) * iterations_per_query}")
    print(f"=" * 70)
    
    results = {
        'test_info': {
            'timestamp': datetime.now().isoformat(),
            'model': tester.model_name,
            'deployment': tester.deployment,
            'endpoint': tester.endpoint,
            'total_queries': len(queries),
            'iterations_per_query': iterations_per_query,
            'total_iterations': len(queries) * iterations_per_query,
            'temperature': 0.3,
            'top_p': 0.950
        },
        'individual_query_results': [],
        'overall_statistics': {},
        'summary_by_query': []
    }
    
    all_eas_scores = []
    query_summaries = []
    
    # Process each query
    for query_idx, query in enumerate(queries, 1):
        print(f"\n🔍 Testing Query {query_idx}/{len(queries)}")
        print(f"📝 Query: {query[:100]}...")
        print(f"🔄 Running {iterations_per_query} iterations...")
        
        query_results = {
            'query_id': query_idx,
            'query_text': query,
            'iterations': [],
            'statistics': {}
        }
        
        query_eas_scores = []
        query_emotion_scores = {emotion: [] for emotion in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]}
        
        # Run iterations for this query
        for iteration in range(1, iterations_per_query + 1):
            print(f"   🔄 Iteration {iteration}/{iterations_per_query}")
            
            emotion_scores = tester.get_single_emotion_scores(query, attempt_number=iteration)
            
            if emotion_scores:
                eas_score = tester.calculate_eas_score(emotion_scores)
                query_eas_scores.append(eas_score)
                all_eas_scores.append(eas_score)
                
                # Store individual emotion scores
                for emotion in query_emotion_scores.keys():
                    if emotion in emotion_scores:
                        query_emotion_scores[emotion].append(emotion_scores[emotion])
                
                # Store iteration result
                iteration_result = {
                    'iteration': iteration,
                    'success': True,
                    'emotion_scores': emotion_scores.copy(),
                    'eas_score': eas_score,
                    'timestamp': datetime.now().isoformat()
                }
                
                query_results['iterations'].append(iteration_result)
                print(f"      ✅ EAS: {eas_score:.6f}")
                
            else:
                iteration_result = {
                    'iteration': iteration,
                    'success': False,
                    'emotion_scores': None,
                    'eas_score': None,
                    'timestamp': datetime.now().isoformat()
                }
                query_results['iterations'].append(iteration_result)
                print(f"      ❌ Failed")
            
            # Wait between iterations
            if iteration < iterations_per_query:
                time.sleep(1)
        
        # Calculate statistics for this query
        if query_eas_scores:
            query_stats = {
                'success_rate': len(query_eas_scores) / iterations_per_query,
                'successful_iterations': len(query_eas_scores),
                'failed_iterations': iterations_per_query - len(query_eas_scores),
                'eas_statistics': {
                    'mean': float(np.mean(query_eas_scores)),
                    'median': float(np.median(query_eas_scores)),
                    'std_deviation': float(np.std(query_eas_scores)),
                    'variance': float(np.var(query_eas_scores)),
                    'min': float(np.min(query_eas_scores)),
                    'max': float(np.max(query_eas_scores)),
                    'range': float(np.max(query_eas_scores) - np.min(query_eas_scores)),
                    'coefficient_of_variation': float(np.std(query_eas_scores) / np.mean(query_eas_scores)) if np.mean(query_eas_scores) > 0 else 0,
                    'all_scores': query_eas_scores.copy()
                },
                'individual_emotion_statistics': {}
            }
            
            # Individual emotion statistics for this query
            for emotion, scores in query_emotion_scores.items():
                if scores:
                    query_stats['individual_emotion_statistics'][emotion] = {
                        'mean': float(np.mean(scores)),
                        'median': float(np.median(scores)),
                        'std_deviation': float(np.std(scores)),
                        'variance': float(np.var(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores)),
                        'range': float(np.max(scores) - np.min(scores)),
                        'coefficient_of_variation': float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0,
                        'all_scores': scores.copy()
                    }
            
            query_results['statistics'] = query_stats
            
            # Create summary for this query
            summary = {
                'query_id': query_idx,
                'query_text': query,
                'mean_eas': query_stats['eas_statistics']['mean'],
                'std_dev_eas': query_stats['eas_statistics']['std_deviation'],
                'cv_eas': query_stats['eas_statistics']['coefficient_of_variation'],
                'success_rate': query_stats['success_rate']
            }
            query_summaries.append(summary)
            
            print(f"   📊 Query Summary:")
            print(f"      Mean EAS: {query_stats['eas_statistics']['mean']:.6f}")
            print(f"      Std Dev: {query_stats['eas_statistics']['std_deviation']:.6f}")
            print(f"      CV: {query_stats['eas_statistics']['coefficient_of_variation']:.4f}")
        
        results['individual_query_results'].append(query_results)
        
        # Brief pause between queries
        if query_idx < len(queries):
            print(f"   ⏳ Waiting 2 seconds before next query...")
            time.sleep(2)
    
    # Calculate overall statistics
    if all_eas_scores:
        # Overall EAS statistics
        results['overall_statistics'] = {
            'total_successful_iterations': len(all_eas_scores),
            'total_possible_iterations': len(queries) * iterations_per_query,
            'overall_success_rate': len(all_eas_scores) / (len(queries) * iterations_per_query),
            'overall_eas_statistics': {
                'mean': float(np.mean(all_eas_scores)),
                'median': float(np.median(all_eas_scores)),
                'std_deviation': float(np.std(all_eas_scores)),
                'variance': float(np.var(all_eas_scores)),
                'min': float(np.min(all_eas_scores)),
                'max': float(np.max(all_eas_scores)),
                'range': float(np.max(all_eas_scores) - np.min(all_eas_scores)),
                'coefficient_of_variation': float(np.std(all_eas_scores) / np.mean(all_eas_scores)) if np.mean(all_eas_scores) > 0 else 0
            },
            'query_level_statistics': {
                'mean_of_query_means': float(np.mean([q['mean_eas'] for q in query_summaries])),
                'std_of_query_means': float(np.std([q['mean_eas'] for q in query_summaries])),
                'mean_of_query_stds': float(np.mean([q['std_dev_eas'] for q in query_summaries])),
                'std_of_query_stds': float(np.std([q['std_dev_eas'] for q in query_summaries])),
                'mean_cv': float(np.mean([q['cv_eas'] for q in query_summaries]))
            }
        }
    
    results['summary_by_query'] = query_summaries
    
    return results

# Legacy function for single query testing - kept for backward compatibility
def print_test_results(results: Dict):
    """
    Print formatted test results (legacy function for single query tests)
    
    Args:
        results: Test results dictionary
    """
    print(f"\n🎯 SINGLE QUERY CONSISTENCY TEST RESULTS")
    print(f"=" * 70)
    
    # Basic info
    print(f"📝 Query: {results['test_info']['query']}")
    print(f"🔄 Iterations: {results['test_info']['iterations']}")
    print(f"✅ Success Rate: {results['statistics']['success_rate']:.1%}")
    
    # EAS Statistics
    eas_stats = results['statistics']['eas_statistics']
    print(f"\n📊 EAS SCORE STATISTICS:")
    print(f"   Mean: {eas_stats['mean']:.6f}")
    print(f"   Std Dev: {eas_stats['std_deviation']:.6f}")
    print(f"   Coefficient of Variation: {eas_stats['coefficient_of_variation']:.4f}")
    print(f"   Range: {eas_stats['min']:.6f} - {eas_stats['max']:.6f}")
    print(f"   Variance: {eas_stats['variance']:.8f}")
    
    # Consistency Analysis
    consistency = results['consistency_analysis']
    print(f"\n🎯 CONSISTENCY ANALYSIS:")
    print(f"   Level: {consistency['consistency_level']}")
    print(f"   CV Interpretation: {consistency['interpretation']['cv_meaning']}")
    
    # Individual emotion statistics
    print(f"\n😊 INDIVIDUAL EMOTION STATISTICS:")
    emotion_stats = results['statistics']['individual_emotion_statistics']
    for emotion, stats in emotion_stats.items():
        print(f"   {emotion.capitalize()}: Mean={stats['mean']:.1f}, Std={stats['std_deviation']:.1f}, CV={stats['coefficient_of_variation']:.3f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in consistency['recommendations']:
        print(f"   • {rec}")

def main():
    """Main execution function"""
    print("🧪 GPT-4o-mini Consistency Test - EAS Score Variance Analysis")
    print("=" * 70)
    
    # Initialize tester
    tester = GPTConsistencyTester()
    
    # Define test query (you can modify this)
def print_multi_query_results(results: Dict):
    """
    Print formatted results for multi-query consistency test
    
    Args:
        results: Multi-query test results dictionary
    """
    print(f"\n🎯 MULTI-QUERY CONSISTENCY TEST RESULTS")
    print(f"=" * 70)
    
    # Basic info
    test_info = results['test_info']
    print(f"📊 Test Overview:")
    print(f"   Total Queries: {test_info['total_queries']}")
    print(f"   Iterations per Query: {test_info['iterations_per_query']}")
    print(f"   Total API Calls: {test_info['total_iterations']}")
    print(f"   Overall Success Rate: {results['overall_statistics']['overall_success_rate']:.1%}")
    
    # Overall statistics
    overall_stats = results['overall_statistics']['overall_eas_statistics']
    print(f"\n📈 OVERALL EAS STATISTICS (All Queries Combined):")
    print(f"   Mean: {overall_stats['mean']:.6f}")
    print(f"   Std Dev: {overall_stats['std_deviation']:.6f}")
    print(f"   Coefficient of Variation: {overall_stats['coefficient_of_variation']:.4f}")
    print(f"   Range: {overall_stats['min']:.6f} - {overall_stats['max']:.6f}")
    
    # Query-level statistics  
    query_stats = results['overall_statistics']['query_level_statistics']
    print(f"\n📊 QUERY-LEVEL STATISTICS:")
    print(f"   Mean of Query Means: {query_stats['mean_of_query_means']:.6f}")
    print(f"   Std of Query Means: {query_stats['std_of_query_means']:.6f}")
    print(f"   Mean of Query Std Devs: {query_stats['mean_of_query_stds']:.6f}")
    print(f"   Std of Query Std Devs: {query_stats['std_of_query_stds']:.6f}")
    print(f"   Average CV across Queries: {query_stats['mean_cv']:.4f}")
    
    # Individual query summaries
    print(f"\n📋 INDIVIDUAL QUERY SUMMARIES:")
    print(f"{'ID':<3} {'Mean EAS':<10} {'Std Dev':<10} {'CV':<8} {'Success':<8} {'Query Preview':<50}")
    print("-" * 90)
    
    for summary in results['summary_by_query']:
        query_preview = summary['query_text'][:47] + "..." if len(summary['query_text']) > 47 else summary['query_text']
        print(f"{summary['query_id']:<3} {summary['mean_eas']:<10.6f} {summary['std_dev_eas']:<10.6f} {summary['cv_eas']:<8.4f} {summary['success_rate']:<8.1%} {query_preview:<50}")

def main():
    """Main execution function"""
    print("🧪 GPT-4o-mini Multi-Query Consistency Test")
    print("📊 Testing 20 Random Queries × 10 Iterations Each")
    print("=" * 70)
    
    # Initialize tester
    tester = GPTConsistencyTester()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load random queries from emotion dataset CSV
    csv_path = "./emotion_dataset/2018_E_c_En_train_emotional.csv"
    
    # Check if file exists, if not try alternative path
    if not os.path.exists(csv_path):
        csv_path = "../EAS_score/emotion_dataset/2018_E_c_En_train_emotional.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found. Please ensure the file exists at:")
        print(f"   {csv_path}")
        return
    
    queries = load_random_queries_from_csv(csv_path, num_queries=20)
    
    if not queries:
        print(f"❌ No queries loaded from CSV file")
        return
    
    try:
        # Run multi-query consistency test
        results = run_multi_query_consistency_test(tester, queries, iterations_per_query=10)
        
        # Print results
        print_multi_query_results(results)
        
        # Save comprehensive results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"gpt4o_multi_query_consistency_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"\n💾 Comprehensive results saved to: {output_file}")
        
        # Create summary CSV for easy analysis
        summary_csv = f"gpt4o_consistency_summary_{timestamp}.csv"
        summary_df = pd.DataFrame(results['summary_by_query'])
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"📊 Summary CSV saved to: {summary_csv}")
        
        # Detailed analysis section
        print(f"\n🔍 DETAILED ANALYSIS SUMMARY")
        print(f"=" * 70)
        
        overall_eas = results['overall_statistics']['overall_eas_statistics']
        query_level = results['overall_statistics']['query_level_statistics']
        
        print(f"📈 Consistency Assessment:")
        overall_cv = overall_eas['coefficient_of_variation']
        avg_query_cv = query_level['mean_cv']
        
        if overall_cv < 0.1 and avg_query_cv < 0.15:
            consistency_assessment = "EXCELLENT - Very consistent across all queries"
        elif overall_cv < 0.2 and avg_query_cv < 0.25:
            consistency_assessment = "GOOD - Reasonably consistent performance"
        elif overall_cv < 0.3 and avg_query_cv < 0.35:
            consistency_assessment = "MODERATE - Some variation in consistency"
        else:
            consistency_assessment = "VARIABLE - High variation in model responses"
        
        print(f"   Assessment: {consistency_assessment}")
        print(f"   Overall CV: {overall_cv:.4f}")
        print(f"   Average Query CV: {avg_query_cv:.4f}")
        print(f"   Query Mean Variation: {query_level['std_of_query_means']:.6f}")
        
        print(f"\n🎉 Multi-query consistency test complete!")
        print(f"📁 Results saved in JSON and CSV formats for detailed analysis")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        # Save partial results if available
        print("💾 Attempting to save partial results...")
    except Exception as e:
        print(f"❌ Error during consistency test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()