#!/bin/bash
# Watch a TC2 training job from the head node and append compact progress
# snapshots to runs/<run>/progress_history.log until the job leaves the queue.
# Runs independently of any SSH session or local machine (launch with nohup):
#
#   cd ~/ai6101-train/safety-model
#   nohup bash cluster/watch_job.sh <jobid> [run_name] >/dev/null 2>&1 < /dev/null &
#
set -u
JOBID="${1:?usage: watch_job.sh <jobid> [run_name]}"
RUN_NAME="${2:-qwen25-1.5b-risk-lora-v1-tc2}"
SM="$HOME/ai6101-train/safety-model"
RUN="$SM/runs/$RUN_NAME"
LOG="$RUN/train_stdout.log"
HIST="$RUN/progress_history.log"
mkdir -p "$RUN"

echo "=== watcher started for job $JOBID at $(date '+%F %T') ===" >> "$HIST"
while [ -n "$(squeue -h -j "$JOBID" 2>/dev/null)" ]; do
  snapshot=$(grep -E '^\[train\] (step|eval@)' "$LOG" 2>/dev/null | tail -1)
  echo "$(date '+%F %T')  ${snapshot:-waiting for log...}" >> "$HIST"
  sleep 60
done
{
  echo "=== job $JOBID left the queue at $(date '+%F %T') ==="
  grep -E '^\[train\] (eval@[0-9]+:|done\.)' "$LOG" 2>/dev/null | tail -8
  [ -f "$RUN/metrics.jsonl" ] && tail -4 "$RUN/metrics.jsonl"
} >> "$HIST"
echo "=== watcher finished at $(date '+%F %T') ===" >> "$HIST"
