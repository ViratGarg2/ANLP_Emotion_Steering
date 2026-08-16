#!/bin/bash

#SBATCH --job-name=conceptor_qwen
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=conceptor_%j.out
#SBATCH --error=conceptor_%j.err

echo "Job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Activate virtual environment
source ~/venv/bin/activate

# Go to project directory
cd ~/qwen_testing/conceptor_antonym/src/functionvectors || exit 1

echo "Working directory: $(pwd)"
echo "Python: $(which python3)"
python3 --version

# Run experiment
python3 run_experiment.py \
    --experiment_type addition,conceptor-paper,conceptor,conceptor-mean \
    --task antonyms \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dtype float16 \
    --mean_activations_path ../../Results/mean_activations_qwen1.5b.pkl \
    --list_extraction_layers 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
    --list_beta_averaging 0.5,1.0,2.0,2.3,3.0,4.0,5.0 \
    --list_beta_conceptor 0.25,0.5,1.0,2.0,3.9,5.0 \
    --list_apertures_normal 0.005,0.05,0.1,1.0,2.0,5.0 \
    --list_apertures_mean_centered 0.005,0.05,0.1,1.0,2.0,5.0 \
    --list_lambda_conceptor 0.25,0.5,0.75,1.0 \
    --num_steering_prompts 100 \
    --num_input_prompts 1000 \
    --num_experiments 1 \
    --batch_size 128 \
    --seed 0 \
    --results_path ../../Results_v3

echo "Experiment finished"
echo "End time: $(date)"