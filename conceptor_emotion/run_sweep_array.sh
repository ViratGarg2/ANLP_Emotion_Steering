#!/bin/bash

#SBATCH --job-name=emo_sweep
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode077,gnode069   # both raise "CUDA unknown error" on init
#SBATCH --constraint=2080ti      # untagged nodes include GTX 1080 Ti (sm_61), unsupported by torch 2.11
#SBATCH --output=logs/emo_sweep_%A_%a.out
#SBATCH --error=logs/emo_sweep_%A_%a.err

# Large 5-emotion sweep: one array task per emotion. The cvit account allows 1 GPU per
# user, so tasks run one after another (~50-60 min each on a 2080 Ti; 615 configs x 50 queries).
#   mkdir -p logs && sbatch -A cvit -p u22 run_sweep_array.sh
# Results: sweep_results_5emo_gpt/<emotion>/{generations,scored,summary}.csv
# If a task times out after generation, rescore with:
#   python3 shift_conceptor_sweep.py --emotions <e> --out_dir sweep_results_5emo_gpt/<e> --score_only

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

# a complete generations.csv (615 configs x 50 queries + 50 baseline rows) only needs scoring
OUT="sweep_results_5emo_gpt/$EMO"
SCORE_ONLY=""
if [ -f "$OUT/generations.csv" ] && [ "$(python3 -c "import pandas as pd; print(len(pd.read_csv('$OUT/generations.csv')))")" -ge 30800 ]; then
    echo "generations complete -> scoring only"; SCORE_ONLY="--score_only"
fi

python3 -u shift_conceptor_sweep.py $SCORE_ONLY \
    --device cuda --dtype float16 \
    --emotions "$EMO" \
    --stats_sources synthetic \
    --n_fit 30 --n_test 50 --n_fit_data 250 \
    --layer_sets 4-10,6-12,8-14,10-16,12-18,14-20,16-22,18-24,8-20,10-14 \
    --shift_lambdas 0.5,1.0,1.5,2.0,2.5 \
    --apertures 0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0 \
    --conc_lambdas 0.1,0.15,0.2,0.25,0.3,0.4,0.5 \
    --ev_alphas 1,2,3,4,5 \
    --batch_size 50 \
    --out_dir "$OUT"

echo "Finished at $(date)"
