#!/bin/bash
# Serve one model and run the Offline scenario (which is also the quality run),
# the scorer, and the Server-scenario search. Same vLLM flags for every model.
#
# usage: scripts/run_serving.sh <model-id-or-path> <prefix> <val|test> <qps-hint> [loadgen args...]
#   results/<prefix>_offline_<split>/  raw.jsonl, offline.json, metrics.json, serving_env.json
#   results/<prefix>_server_<split>/   server.json
# env: SKIP_SERVER=1 to run only Offline + scoring; SKIP_OFFLINE=1 for Server only; OFFLINE_CONCURRENCY (default 256)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; PREFIX="$2"; SPLIT="$3"; QPS_HINT="$4"; shift 4
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_OFFLINE=1
PY=/workspace/venv-serve/bin/python
OFFLINE="${PREFIX}_offline_${SPLIT}"
SERVER="${PREFIX}_server_${SPLIT}"

scripts/serve_vllm.sh "$MODEL" "$PREFIX"
PID=$(cat /workspace/logs/vllm.pid)
LOG="/workspace/logs/vllm_${PREFIX}.log"
trap 'kill $PID 2>/dev/null || true; sleep 8' EXIT
for _ in $(seq 1 360); do
  curl -sf localhost:8000/v1/models > /dev/null && break
  kill -0 "$PID" 2>/dev/null || { echo "vLLM died"; tail -30 "$LOG"; exit 1; }
  sleep 5
done
curl -sf localhost:8000/v1/models > /dev/null || { echo "vLLM not ready"; exit 1; }

if [ "${SKIP_OFFLINE:-0}" != "1" ]; then
$PY -m src.loadgen offline --name "$OFFLINE" --split "$SPLIT" --model "$PREFIX" \
  --concurrency "${OFFLINE_CONCURRENCY:-256}" "$@" 2>&1 | grep -v "Warning\|cached" | tail -25
$PY -m src.score "results/$OFFLINE/raw.jsonl" 2> /dev/null | head -9

# Serving environment: versions, GPU, vLLM command and its reported memory use.
$PY - "$OFFLINE" "$LOG" <<'PYEOF'
import json, re, subprocess, sys, importlib.metadata as md
name, log = sys.argv[1], sys.argv[2]
text = open(log, errors="ignore").read()
def grab(pattern):
    m = re.search(pattern, text)
    return m.group(1) if m else None
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
info = {
    "vllm": md.version("vllm"), "torch": md.version("torch"),
    "transformers": md.version("transformers"), "compressed_tensors": md.version("compressed-tensors"),
    "gpu": gpu,
    "vllm_command": open(log + ".cmd").read().strip(),
    "model_weights_gib": grab(r"Model loading took ([\d.]+) GiB"),
    "kv_cache_tokens": grab(r"GPU KV cache size: ([\d,]+) tokens"),
}
json.dump(info, open(f"results/{name}/serving_env.json", "w"), indent=2)
print(json.dumps(info))
PYEOF

fi

if [ "${SKIP_SERVER:-0}" != "1" ]; then
  # SERVER_QPS set: validate that fixed rate twice (no search); otherwise search from the hint.
  if [ -n "${SERVER_QPS:-}" ]; then MODE=(--qps "$SERVER_QPS" --confirm); else MODE=(--search --confirm --qps-hint "$QPS_HINT"); fi
  $PY -m src.loadgen server --name "$SERVER" --split "$SPLIT" --model "$PREFIX" \
    "${MODE[@]}" "$@" 2>&1 | grep -v "Warning\|cached" | grep -E "^probe|max_valid_qps|prepared"
fi
echo "RUN_SERVING DONE $PREFIX $SPLIT"
