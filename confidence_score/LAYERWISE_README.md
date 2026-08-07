# Layerwise Confidence Score Evaluator

## Overview

The confidence score evaluator now supports **layerwise analysis** - testing individual layers of emotion vectors to understand how different transformer layers contribute to emotional steering.

## New Layerwise Mode

### What it does:
- Tests each layer's emotion vector **individually** (not combined)
- Uses the first 25 queries from EmotionQuery+.json (configurable)  
- Focuses on a single target emotion (default: joy)
- Measures confidence change for each layer separately

### Key Features:
- **Layer-by-layer analysis**: See which layers are most effective
- **Detailed CSV output**: Every response with confidence scores
- **JSON summary**: Statistical analysis across all layers
- **Fair comparison**: Same queries used for all layers

## Usage

### Basic Layerwise Evaluation
```bash
python confidence_score_evaluator.py --layerwise --alpha 2.0
```

### Custom Parameters  
```bash
python confidence_score_evaluator.py \
  --layerwise \
  --alpha 2.0 \
  --target_emotion joy \
  --num_queries 25 \
  --vectors_dir ./emotion_vectors \
  --emotion_queries_file EmotionQuery+.json
```

### Standard Mode (All Layers Combined)
```bash
python confidence_score_evaluator.py \
  --alpha 2.0 \
  --num_queries 100 \
  --emotion all
```

## Arguments

### Layerwise Mode Arguments
- `--layerwise`: Enable layerwise analysis (test each layer individually)
- `--target_emotion`: Target emotion for layerwise analysis (default: joy)
- `--num_queries`: Number of queries to test (default: 25)
- `--alpha`: Emotion vector strength multiplier (required)

### General Arguments  
- `--vectors_dir`: Directory with emotion vector JSON files
- `--emotion_queries_file`: Path to EmotionQuery+.json
- `--output_file`: Custom output CSV filename
- `--model_name`: HuggingFace model (default: Qwen/Qwen2.5-1.5B-Instruct)
- `--device`: Device to run on (cuda/cpu)

## Outputs

### CSV File (Layerwise)
Contains for each layer and query:
- `emotion`: Target emotion being tested
- `layer`: Layer number being tested
- `query_idx`: Query index (0-24)
- `alpha`: Alpha multiplier used
- `baseline_target_confidence`: Confidence without EV injection
- `enhanced_target_confidence`: Confidence with EV injection  
- `confidence_change`: Difference (enhanced - baseline)
- `query`: Input query text
- `baseline_response`: Response without emotion steering
- `enhanced_response`: Response with emotion steering
- `baseline_confidence_all`: All emotion confidences (JSON)
- `enhanced_confidence_all`: All emotion confidences (JSON)
- `timestamp`: When evaluation was performed

### JSON Summary (Layerwise)
Statistical analysis including:
- **Overall summary**: Best/worst/most effective layers
- **Per-layer statistics**: Mean, std, min, max confidence changes
- **Effect analysis**: Positive/negative/zero change counts
- **Methodology**: How analysis was performed

## Example Results Interpretation

| Layer | Avg Change | Positive Rate | Effect Strength | Interpretation |
|-------|------------|---------------|-----------------|----------------|
| 15    | +0.234     | 80%          | 0.298          | Highly effective |
| 8     | +0.156     | 64%          | 0.189          | Moderately effective |
| 3     | +0.023     | 52%          | 0.087          | Minimally effective |
| 22    | -0.045     | 36%          | 0.112          | Counterproductive |

### Insights:
- **Best layer**: Highest average confidence change
- **Most effective**: Highest effect strength (absolute change)
- **Most consistent**: Highest positive rate

## Technical Details

- **Model**: Qwen/Qwen2.5-1.5B-Instruct for generation
- **Classifier**: BART-large-MNLI for confidence scoring  
- **Vector injection**: Forward hooks with L2 normalization
- **Query selection**: Sequential (first N) for layerwise analysis
- **Evaluation**: Each layer tested independently

## File Requirements

1. **Emotion Vectors**: JSON files in format `Qwen25_1B5_{emotion}_vectors.json`
2. **Query Dataset**: EmotionQuery+.json with list of queries
3. **Python Environment**: transformers, torch, numpy, tqdm

## Quick Start

1. **Check setup**: `python layerwise_example.py`
2. **Run evaluation**: `python confidence_score_evaluator.py --layerwise --alpha 2.0`
3. **Analyze results**: Check generated CSV and JSON files

## Comparison Modes

| Mode | Layers | Queries | Purpose |
|------|--------|---------|---------|
| Standard | All combined | Random sample | Overall EV effectiveness |
| Layerwise | Individual | Sequential | Layer-specific analysis |

The layerwise mode is perfect for understanding **which layers** contribute most to emotional steering, while standard mode measures **overall effectiveness** across all layers combined.