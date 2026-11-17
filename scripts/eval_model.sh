#!/bin/bash
# Serve one model with vLLM, run inference on a split, score it, stop the server.
#
# usage: scripts/eval_model.sh <model-id-or-path> <results-name> <split> [extra infer args...]
# env:   GPU_FRAC (vLLM --gpu-memory-utilization, default 0.9)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; NAME="$2"; SPLIT="$3"; shift 3
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_OFFLINE=1
PY=/workspace/venv-serve/bin/python

EXTRA_FLAGS="--gpu-memory-utilization=${GPU_FRAC:-0.9}" scripts/serve_vllm.sh "$MODEL" "$NAME"
PID=$(cat /workspace/logs/vllm.pid)
trap 'kill $PID 2>/dev/null || true; sleep 5' EXIT
for _ in $(seq 1 120); do
  curl -sf localhost:8000/v1/models > /dev/null && break
  kill -0 "$PID" 2>/dev/null || { echo "vLLM died"; tail -20 "/workspace/logs/vllm_$NAME.log"; exit 1; }
  sleep 5
done
curl -sf localhost:8000/v1/models > /dev/null || { echo "vLLM not ready"; exit 1; }

$PY -m src.infer --name "$NAME" --split "$SPLIT" --model "$NAME" "$@" 2>&1 | grep -v "Warning\|cached" | tail -2
$PY -m src.score "results/$NAME/raw.jsonl" 2> /dev/null | head -9
