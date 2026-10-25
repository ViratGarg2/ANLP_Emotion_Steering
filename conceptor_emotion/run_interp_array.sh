#!/bin/bash

#SBATCH --job-name=interp_conc
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode077,gnode069   # both raise "CUDA unknown error" on init
#SBATCH --constraint=2080ti      # untagged nodes include GTX 1080 Ti (sm_61), unsupported by torch 2.11
#SBATCH --output=logs/interp_conc_%A_%a.out
#SBATCH --error=logs/interp_conc_%A_%a.err

# Interpolation protocol of Triantafyllopoulos et al. (arXiv:2605.04980) on emotion:
# z' = (1 - beta) z + beta C z, all-token steering, single layers, wide aperture range,
# fitted on the gpt-5.5 pairs (324 configs x 50 queries per emotion; tasks run one after
# another). Queue behind the published-rule run:
#   sbatch -A cvit -p u22 --dependency=afterany:<paper_array_id> run_interp_array.sh
# Results: sweep_results_5emo_interp/<emotion>/{generations,scored,summary}.csv

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

# a complete generations.csv (324 configs x 50 queries + 50 baseline rows) only needs scoring
OUT="sweep_results_5emo_interp/$EMO"
SCORE_ONLY=""
if [ -f "$OUT/generations.csv" ] && [ "$(python3 -c "import pandas as pd; print(len(pd.read_csv('$OUT/generations.csv')))")" -ge 16250 ]; then
    echo "generations complete -> scoring only"; SCORE_ONLY="--score_only"
fi

python3 -u shift_conceptor_sweep.py $SCORE_ONLY \
    --device cuda --dtype float16 \
    --emotions "$EMO" \
    --stats_sources synthetic \
    --n_fit 30 --n_test 50 --n_fit_data 250 \
    --methods conceptor_interp --token_scope all \
    --layer_sets 4-4,8-8,10-10,12-12,14-14,16-16,18-18,20-20,24-24 \
    --apertures 0.01,0.05,0.1,0.2,0.5,1,2,5,10 \
    --interp_betas 0.2,0.4,0.6,0.8 \
    --batch_size 50 \
    --out_dir "$OUT"

echo "Finished at $(date)"
