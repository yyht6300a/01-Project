#!/bin/bash
# One-shot status/report for the cloud training runs.
# Usage (from anywhere with campus network):
#   ssh fanlong001@10.96.189.12 'bash ~/ai6101-train/safety-model/cluster/tc2_status.sh'
set -u
SM="$HOME/ai6101-train/safety-model"
echo "== jobs ($(date '+%F %T')) =="
squeue -u "$(whoami)" || true
for run in "$SM"/runs/*/; do
  [ -d "$run" ] || continue
  [ -f "$run/train_stdout.log" ] || continue
  echo
  echo "== $(basename "$run") =="
  echo "-- last log lines --"
  tail -3 "$run/train_stdout.log"
  echo "-- evals --"
  if [ -f "$run/metrics.jsonl" ]; then cat "$run/metrics.jsonl"; else echo "(no metrics yet)"; fi
  echo "-- progress tail --"
  if [ -f "$run/progress_history.log" ]; then tail -6 "$run/progress_history.log"; else echo "(no watcher history)"; fi
  echo "-- checkpoints --"
  if ls "$run/best" >/dev/null 2>&1; then echo "best/: $(ls -1 "$run/best" | wc -l) files"; else echo "best/: none"; fi
  if ls "$run/last" >/dev/null 2>&1; then echo "last/: $(ls -1 "$run/last" | wc -l) files"; else echo "last/: none"; fi
done
