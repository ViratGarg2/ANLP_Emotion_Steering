#!/bin/bash

#SBATCH --job-name=paper_conc
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode077,gnode069   # both raise "CUDA unknown error" on init
#SBATCH --constraint=2080ti      # untagged nodes include GTX 1080 Ti (sm_61), unsupported by torch 2.11
#SBATCH --output=logs/paper_conc_%A_%a.out
#SBATCH --error=logs/paper_conc_%A_%a.err

# Published conceptor rule h' = beta * C h (Postmus & Abreu) on emotion, fitted on the
# gpt-5.5 pairs: single layers and two late blocks x aperture x beta (294 configs x 50
# queries per emotion, ~20-25 min each on a 2080 Ti; tasks run one after another).
#   mkdir -p logs && sbatch -A cvit -p u22 run_paper_array.sh
# Results: sweep_results_5emo_paper/<emotion>/{generations,scored,summary}.csv

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

# a complete generations.csv (294 configs x 50 queries + 50 baseline rows) only needs scoring
OUT="sweep_results_5emo_paper/$EMO"
SCORE_ONLY=""
if [ -f "$OUT/generations.csv" ] && [ "$(python3 -c "import pandas as pd; print(len(pd.read_csv('$OUT/generations.csv')))")" -ge 14750 ]; then
    echo "generations complete -> scoring only"; SCORE_ONLY="--score_only"
fi

python3 -u shift_conceptor_sweep.py $SCORE_ONLY \
    --device cuda --dtype float16 \
    --emotions "$EMO" \
    --stats_sources synthetic \
    --n_fit 30 --n_test 50 --n_fit_data 250 \
    --methods conceptor_paper \
    --layer_sets 12-12,16-16,20-20,24-24,16-22,18-24 \
    --apertures 0.05,0.1,0.2,0.5,1,2,5 \
    --paper_betas 0.5,0.75,1,1.25,1.5,2,4 \
    --batch_size 50 \
    --out_dir "$OUT"

echo "Finished at $(date)"
