#!/bin/bash
# One-shot TC2 bootstrap for the school-GPU LoRA run.
# Run on the HEAD NODE (safe under nohup; survives SSH / campus-network drops):
#
#   cd ~/ai6101-train/safety-model && nohup bash cluster/bootstrap_tc2.sh > ~/ai6101-train/bootstrap.log 2>&1 &
#
# Steps (idempotent, each skipped if already done):
#   1. conda env "ai6101" (python 3.11)
#   2. torch (cu124) + transformers/peft/... into that env
#   3. download Qwen2.5-1.5B-Instruct into ~/ai6101-train/models
#   4. sanity checks, then sbatch cluster/submit_train.sbatch
set -euo pipefail

export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
BASE="$HOME/ai6101-train"
SM="$BASE/safety-model"
CONDA=/apps/anaconda3-2025.06-0/bin/conda
ENVP="$HOME/.conda/envs/ai6101"
HF="$HOME/.local/bin/hf"
SBATCH=/cm/local/apps/slurm/current/bin/sbatch
MODEL="$BASE/models/Qwen2.5-1.5B-Instruct"

cd "$SM"
echo "[bootstrap] start $(date) on $(hostname)"
wc -l data_v1/train.jsonl data_v1/val.jsonl

# 1) conda env
if [ ! -x "$ENVP/bin/python" ]; then
  echo "[bootstrap] creating conda env ai6101 (python 3.11)"
  "$CONDA" create -y -q -n ai6101 python=3.11
else
  echo "[bootstrap] conda env already present"
fi
PY="$ENVP/bin/python"
PIP="$ENVP/bin/pip"

# 2) pip deps (torch cu124 ~2.5GB download: the slow part)
echo "[bootstrap] pip deps: $(date)"
"$PIP" install -q --upgrade pip
"$PIP" install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$PIP" install -q "transformers>=4.56" "peft>=0.14" "accelerate>=0.34" "huggingface_hub>=0.26" "PyYAML>=6" "numpy>=1.26"
"$PY" - <<'PYEOF'
import torch, transformers, peft
print("[bootstrap] torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
PYEOF

# 3) base model
mkdir -p "$BASE/models"
if [ ! -f "$MODEL/config.json" ] || ! ls "$MODEL"/*.safetensors >/dev/null 2>&1; then
  echo "[bootstrap] downloading Qwen/Qwen2.5-1.5B-Instruct (~3.1GB): $(date)"
  "$HF" download Qwen/Qwen2.5-1.5B-Instruct --local-dir "$MODEL"
else
  echo "[bootstrap] base model already present"
fi
ls "$MODEL"/*.safetensors >/dev/null 2>&1 || { echo "[bootstrap] ERROR: model weights missing"; exit 1; }

# 4) data sanity
"$PY" - <<'PYEOF'
import json
train = sum(1 for _ in open("data_v1/train.jsonl", encoding="utf-8"))
val = sum(1 for _ in open("data_v1/val.jsonl", encoding="utf-8"))
print(f"[bootstrap] data_v1 train={train} val={val}")
assert train > 4000 and val > 600, "unexpected split sizes"
PYEOF

# 5) submit the training job
mkdir -p logs runs
JOBID=$("$SBATCH" --parsable cluster/submit_train.sbatch)
echo "$JOBID" > logs/last_submitted_job.txt
echo "[bootstrap] submitted job $JOBID: $(date)"
echo "[bootstrap] follow: tail -f runs/qwen25-1.5b-risk-lora-v1-tc2/train_stdout.log"
