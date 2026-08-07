# EAS Score Evaluator

## Overview

This script evaluates the effectiveness of emotion vectors using the Emotion Appropriateness Score (EAS) methodology. The EAS score measures how much emotion vector steering improves (or degrades) the emotional appropriateness of generated text.

## EAS Methodology

### Formula
```
EAS = Score(steered_response) - Score(baseline_response)
```

### Scoring Details
- **Individual Emotion Scores**: 0.0 to 1.0 (how well a response expresses the target emotion)
- **EAS Scores**: -1.0 to +1.0 (difference between steered and baseline responses)

### Interpretation
- **Positive EAS** (+0.1 to +1.0): Emotion vector improved emotional appropriateness
- **Zero EAS** (0.0): No change in emotional appropriateness  
- **Negative EAS** (-0.1 to -1.0): Emotion vector decreased emotional appropriateness

## Usage

### Basic Usage
```bash
python eas_score_evaluator.py --alpha 2.0 --num_queries 100 --api_key YOUR_API_KEY
```

### Evaluate Specific Emotion
```bash
python eas_score_evaluator.py --alpha 1.5 --emotion joy --num_queries 50 --api_key YOUR_API_KEY
```

### Arguments
- `--alpha`: Multiplier for emotion vector strength (required)
- `--api_key`: Azure OpenAI API key for GPT-4o-mini (required)
- `--num_queries`: Number of queries to evaluate (default: 50)
- `--emotion`: Specific emotion to evaluate [joy|anger|disgust|fear|sadness|all] (default: all)
- `--vectors_dir`: Directory containing emotion vector JSON files (default: ./emotion_vectors)
- `--emotion_queries_file`: Path to emotion queries JSON file (default: EmotionQuery+.json)

## Outputs

### CSV File
Contains detailed results for each evaluation:
- `emotion`: Target emotion being evaluated
- `query_idx`: Index of the query
- `alpha`: Emotion vector multiplier used
- `baseline_emotion_score`: Emotion score for unsteered response (0.0-1.0)
- `enhanced_emotion_score`: Emotion score for steered response (0.0-1.0) 
- `eas_score`: EAS score (steered - baseline) (-1.0 to +1.0)
- `query`: Input query text
- `baseline_response`: Generated response without emotion steering
- `enhanced_response`: Generated response with emotion steering
- `timestamp`: When the evaluation was performed

### JSON Summary File
Contains statistical analysis:
- Overall statistics (mean, std, min, max EAS scores)
- Per-emotion statistics
- Count of positive/negative/zero EAS scores
- Evaluation metadata and methodology notes

## Example Results Interpretation

| Baseline Score | Steered Score | EAS Score | Interpretation |
|---------------|---------------|-----------|----------------|
| 0.2 | 0.8 | +0.6 | Strong improvement |
| 0.5 | 0.7 | +0.2 | Moderate improvement |
| 0.6 | 0.6 | 0.0 | No change |
| 0.8 | 0.4 | -0.4 | Strong degradation |

## Requirements

- Python 3.7+
- PyTorch
- Transformers 
- OpenAI Python library
- Azure OpenAI API access
- Pre-extracted emotion vectors (JSON format)
- Qwen/Qwen2.5-1.5B-Instruct model

## Files Required

1. **Emotion Vectors**: JSON files in `vectors_dir` named `Qwen25_1B5_{emotion}_vectors.json`
2. **Query Dataset**: JSON file with evaluation queries (list format)
3. **API Key**: Valid Azure OpenAI API key for GPT-4o-mini access

## Technical Details

- **Generation Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Evaluation Model**: GPT-4o-mini via Azure OpenAI
- **Emotion Vector Injection**: Forward hooks with L2 normalization
- **Temperature**: 0.7 for generation, 0.1 for evaluation (consistent scoring)
- **Rate Limiting**: 0.1s delay between API calls