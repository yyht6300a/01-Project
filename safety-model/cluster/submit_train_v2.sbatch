#!/bin/bash
#SBATCH --job-name=ai6101-sft-v2
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/sft-v2-%j.out
#SBATCH --error=logs/sft-v2-%j.err

# LoRA SFT v2 (format v2, data_v2 group-isolated split) on one TC2 A40.
set -u -o pipefail
cd "$SLURM_SUBMIT_DIR"
RUN_DIR="runs/qwen25-1.5b-risk-lora-v2-tc2"
LOG="$RUN_DIR/train_stdout.log"
mkdir -p logs "$RUN_DIR"

PY="$HOME/.conda/envs/ai6101/bin/python"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME=""
if [ -f "$RUN_DIR/last/adapter_config.json" ]; then
  RESUME="--resume"
  echo "[sbatch] found last checkpoint; will resume from it"
fi

echo "[sbatch] job $SLURM_JOB_ID on $(hostname) start: $(date)"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu,driver_version --format=csv,noheader

# Cloud-side progress watcher: lives inside this job allocation, so progress is
# recorded even when nobody is connected. Killed when the training exits.
( while sleep 60; do
    snapshot=$(grep -E '^\[train\] (step|eval@)' "$LOG" 2>/dev/null | tail -1)
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
    echo "$(date '+%F %T')  ${snapshot:-waiting for log...}  gpu=${gpu:-n/a}" >> "$RUN_DIR/progress_history.log"
  done ) &
WATCHER_PID=$!

rc=0
if ! "$PY" -u train_sft.py --config configs/sft_lora_v2_tc2.yaml $RESUME 2>&1 | tee -a "$LOG"; then
  rc=1
fi
kill "$WATCHER_PID" 2>/dev/null || true
{
  echo "=== training exited rc=$rc at $(date '+%F %T') ==="
  grep -E '^\[train\] (eval@[0-9]+:|done\.)' "$LOG" 2>/dev/null | tail -8
} >> "$RUN_DIR/progress_history.log"
echo "[sbatch] training exited rc=$rc at $(date)"
ls "$RUN_DIR" 2>/dev/null || true
exit "$rc"
