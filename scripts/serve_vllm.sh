#!/bin/bash
# Start vLLM for one model with the launch flags from configs/base.yaml.
# The flags are identical for every row of the main table.
#
# usage: scripts/serve_vllm.sh <model-id-or-path> [served-model-name]
# env:   VENV (default /workspace/venv-serve), PORT (default 8000), LOG_DIR
set -euo pipefail

MODEL="$1"
SERVED_NAME="${2:-$1}"
VENV="${VENV:-/workspace/venv-serve}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/vllm_$(echo "$SERVED_NAME" | tr '/ ' '__').log"

FLAGS=(
  --max-model-len=32768
  --limit-mm-per-prompt.video=0
  --no-enable-prefix-caching
)

echo "vllm serve $MODEL --served-model-name $SERVED_NAME ${FLAGS[*]}" > "$LOG.cmd"
nohup "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  "${FLAGS[@]}" > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$LOG_DIR/vllm.pid"
echo "started vllm pid $(cat "$LOG_DIR/vllm.pid"), log $LOG"
