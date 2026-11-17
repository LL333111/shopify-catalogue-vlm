#!/bin/bash
# Choose the Offline client concurrency on val: serve one model and time a fixed
# 600-example subset at several concurrencies. The chosen value is then fixed
# for every row of the table.
#
# usage: scripts/offline_sweep.sh <model-path> <served-name> "<concurrencies>"
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; NAME="$2"; LEVELS="$3"
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_OFFLINE=1
PY=/workspace/venv-serve/bin/python

scripts/serve_vllm.sh "$MODEL" "$NAME"
PID=$(cat /workspace/logs/vllm.pid)
trap 'kill $PID 2>/dev/null || true; sleep 8' EXIT
for _ in $(seq 1 180); do
  curl -sf localhost:8000/v1/models > /dev/null && break
  kill -0 "$PID" 2>/dev/null || { echo "vLLM died"; exit 1; }
  sleep 5
done
for C in $LEVELS; do
  rm -rf "results/sweep_c${C}_offline_val"
  $PY -m src.loadgen offline --name "sweep_c${C}_offline_val" --split val --model "$NAME" \
    --concurrency "$C" --limit 600 > /dev/null 2>&1
  $PY -c "import json;d=json.load(open('results/sweep_c${C}_offline_val/offline.json'));print('concurrency', $C, 'samples_per_s', round(d['samples_per_s'],3), 'errors', d['errors'], 'p99', round(d['latency_p99_s'],1))"
done
echo "SWEEP DONE"
