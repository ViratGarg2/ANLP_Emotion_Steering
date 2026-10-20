#!/bin/bash

#SBATCH --job-name=shift_conc_sweep
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err

# Usage (from qwen_testing/conceptor_emotion):
#   sbatch run_sweep.sh                       # joy,sadness
#   EMOTIONS=joy,anger,disgust,fear,sadness sbatch run_sweep.sh

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/venv/bin/activate
cd ~/qwen_testing/conceptor_emotion || exit 1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # models are pre-cached; avoids hub calls with a stale token

# fail fast on nodes whose GPU cannot run this torch build (gnode077 raised "CUDA unknown error")
python3 -c "import torch; x = torch.randn(64, 64, device='cuda', dtype=torch.float16); print('cuda ok', torch.cuda.get_device_name(), (x @ x).float().abs().sum().item() > 0)" || exit 1

python3 -u shift_conceptor_sweep.py \
    --device cuda --dtype float16 \
    --emotions "${EMOTIONS:-joy,sadness}" \
    --stats_sources model_gen,synthetic,goemotions \
    --n_fit 30 --n_test 20 --n_fit_data 200 \
    --batch_size 20 \
    --out_dir "sweep_results_${EMOTIONS_TAG:-2emo}"

echo "Finished at $(date)"
