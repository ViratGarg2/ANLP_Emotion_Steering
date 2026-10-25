#!/bin/bash

#SBATCH --job-name=bch_layers
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode077,gnode069   # both raise "CUDA unknown error" on init
#SBATCH --constraint=2080ti      # untagged nodes include GTX 1080 Ti (sm_61), unsupported by torch 2.11
#SBATCH --output=logs/bch_layers_%A_%a.out
#SBATCH --error=logs/bch_layers_%A_%a.err

# Benchmark selection sweep: published conceptor rule h' = beta * C h at EVERY single layer
# (0-27; layers 12/16/20/24 are already in sweep_results_5emo_paper), so the best single
# layer can be picked per emotion and compared against stacked shift+conceptor on held-out
# prompts (benchmark_test.py). 24 layers x 5 apertures x 5 betas x 50 queries per emotion.
#   sbatch -A cvit -p u22 --dependency=afterany:<previous_array_id> run_bch_layers_array.sh
# Results: sweep_results_5emo_bch_layers/<emotion>/{generations,scored,summary}.csv

EMOS=(joy anger disgust fear sadness)
EMO=${EMOS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID ($EMO), job $SLURM_ARRAY_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/venv/bin/activate
cd ~/qwen_testing/conceptor_emotion || exit 1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # models are pre-cached; avoids hub calls with a stale token
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -c "import torch; x = torch.randn(64, 64, device='cuda', dtype=torch.float16); print('cuda ok', torch.cuda.get_device_name(), (x @ x).float().abs().sum().item() > 0)" || exit 1

# a complete generations.csv (600 configs x 50 queries + 50 baseline rows) only needs scoring
OUT="sweep_results_5emo_bch_layers/$EMO"
SCORE_ONLY=""
if [ -f "$OUT/generations.csv" ] && [ "$(python3 -c "import pandas as pd; print(len(pd.read_csv('$OUT/generations.csv')))")" -ge 30050 ]; then
    echo "generations complete -> scoring only"; SCORE_ONLY="--score_only"
fi

python3 -u shift_conceptor_sweep.py $SCORE_ONLY \
    --device cuda --dtype float16 \
    --emotions "$EMO" \
    --stats_sources synthetic \
    --n_fit 30 --n_test 50 --n_fit_data 250 \
    --methods conceptor_paper \
    --layer_sets 0-0,1-1,2-2,3-3,4-4,5-5,6-6,7-7,8-8,9-9,10-10,11-11,13-13,14-14,15-15,17-17,18-18,19-19,21-21,22-22,23-23,25-25,26-26,27-27 \
    --apertures 0.1,0.2,0.5,1,2 \
    --paper_betas 1,1.25,1.5,2,4 \
    --batch_size 50 \
    --out_dir "$OUT"

echo "Finished at $(date)"
