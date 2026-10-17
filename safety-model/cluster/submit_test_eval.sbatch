#!/bin/bash
#SBATCH --job-name=ai6101-test-eval
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/testeval-%j.out
#SBATCH --error=logs/testeval-%j.err

# Held-out evaluation on data_v3/test.jsonl for one adapter.
# Usage: sbatch cluster/submit_test_eval.sbatch <adapter-dir> <target-format>
set -u -o pipefail
cd "$SLURM_SUBMIT_DIR"
ADAPTER_DIR="${1:?adapter dir required}"
TARGET_FORMAT="${2:-4field}"
PY="$HOME/.conda/envs/ai6101/bin/python"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[sbatch] job $SLURM_JOB_ID on $(hostname) start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -u test_eval.py --data data_v3/test.jsonl \
  --model /home/msai/fanlong001/ai6101-train/models/Qwen2.5-1.5B-Instruct \
  --adapter "$ADAPTER_DIR" --target-format "$TARGET_FORMAT"
echo "[sbatch] test eval done rc=$?"
