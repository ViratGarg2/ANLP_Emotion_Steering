#!/bin/bash

#SBATCH --job-name=stacked_fv
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --exclude=gnode077,gnode069   # both raise "CUDA unknown error" on init
#SBATCH --constraint=2080ti      # untagged nodes include GTX 1080 Ti (sm_61), unsupported by torch 2.11
#SBATCH --output=logs/stacked_fv_%A_%a.out
#SBATCH --error=logs/stacked_fv_%A_%a.err

# Stacked shift+conceptor vs single-layer addition / published conceptor on the single-token
# function tasks, all on one prompt draw (seed 0, 100 ICL prompts, 1000 zero-shot inputs).
#   mkdir -p logs && sbatch -A cvit -p u22 run_stacked_tasks.sh
#   QUICK=--quick sbatch -A cvit -p u22 --array=0 run_stacked_tasks.sh     # smoke test
# Results: Results_stacked/Qwen2.5-1.5B-Instruct_<task>_stacked.csv

TASKS=(antonyms capitalize)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
echo "Task $TASK, job $SLURM_ARRAY_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/venv/bin/activate
cd ~/qwen_testing/conceptor_antonym/src/functionvectors || exit 1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -c "import torch; x = torch.randn(64, 64, device='cuda', dtype=torch.float16); print('cuda ok', torch.cuda.get_device_name(), (x @ x).float().abs().sum().item() > 0)" || exit 1

python3 -u stacked_experiment.py --task "$TASK" --results_path ../../Results_stacked $QUICK

echo "Finished at $(date)"
