#!/usr/bin/env python3
"""
Example script showing how to run layerwise confidence evaluation

This script demonstrates the new layerwise analysis feature that tests
each layer individually for a specific emotion (default: joy).
"""

import subprocess
import sys
import os

def run_layerwise_evaluation():
    """Run layerwise evaluation with example parameters"""
    
    print("Running Layerwise Emotion Vector Evaluation Example")
    print("=" * 60)
    print("Configuration:")
    print("  • Target emotion: joy")
    print("  • Alpha multiplier: 2.0")
    print("  • Number of queries: 25 (first 25 from EmotionQuery+.json)")
    print("  • Mode: Test each layer individually")
    print("=" * 60)
    
    # Example command for layerwise evaluation
    cmd = [
        sys.executable, "confidence_score_evaluator.py",
        "--layerwise",
        "--alpha", "2.0",
        "--target_emotion", "joy", 
        "--num_queries", "25",
        "--vectors_dir", "./emotion_vectors",
        "--emotion_queries_file", "EmotionQuery+.json"
    ]
    
    print("Command to run:")
    print(" ".join(cmd))
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "confidence_score_evaluator.py",
        "EmotionQuery+.json",
        "./emotion_vectors"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files/directories:")
        for file in missing_files:
            print(f"   • {file}")
        print("\nPlease ensure all required files are available before running.")
        return False
    
    print("✅ All required files found!")
    print("\nTo run the layerwise evaluation, execute:")
    print(f"python {' '.join(cmd[1:])}")
    print("\nThis will:")
    print("1. Load joy emotion vectors from ./emotion_vectors/")
    print("2. Test each layer individually on first 25 queries")
    print("3. Measure confidence changes due to each layer's contribution")
    print("4. Generate detailed CSV with all results")
    print("5. Create JSON summary with layer-wise statistics")
    
    return True

if __name__ == "__main__":
    success = run_layerwise_evaluation()
    if success:
        print("\n🚀 Ready to run layerwise evaluation!")
    else:
        print("\n❌ Setup incomplete")
        sys.exit(1)