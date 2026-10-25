#!/bin/bash

#SBATCH --job-name=benchmark
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --exclude=gnode077,gnode069   # both raise "CUDA unknown error" on init
#SBATCH --constraint=2080ti      # untagged nodes include GTX 1080 Ti (sm_61), unsupported by torch 2.11
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# Held-out benchmark: best stacked shift+conceptor vs best single-layer published conceptor
# (and plain shift / project EV), selected on the sweep prompts, tested on 100 unused prompts.
# Queue behind the per-layer beta*C*h sweep:
#   sbatch -A cvit -p u22 --dependency=afterany:<bch_layers_array_id> run_benchmark.sh
# Results: benchmark/selection.json, benchmark/test/{scored,summary,report_table,report_paired}.csv

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/venv/bin/activate
cd ~/qwen_testing/conceptor_emotion || exit 1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # models are pre-cached; avoids hub calls with a stale token
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -c "import torch; x = torch.randn(64, 64, device='cuda', dtype=torch.float16); print('cuda ok', torch.cuda.get_device_name(), (x @ x).float().abs().sum().item() > 0)" || exit 1

python3 -u benchmark_test.py select --out benchmark/selection.json || exit 1
python3 -u benchmark_test.py run --selection benchmark/selection.json --out_dir benchmark/test \
    --device cuda --dtype float16 --n_test 100 --n_fit_data 250 --batch_size 50 || exit 1
python3 -u benchmark_test.py report --out_dir benchmark/test

echo "Finished at $(date)"
