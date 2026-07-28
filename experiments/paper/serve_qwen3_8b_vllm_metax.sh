#!/usr/bin/env bash
set -eu

python_bin=${PYTHON_BIN:-/mnt/geogpt-doc-new/deepresearch/gepa-multi-skill/.venv/bin/python}
model_dir=${QWEN3_8B_MODEL_DIR:-/mnt/geogpt-doc-new/deepresearch/tool_jepa_qwen35_9b/models/Qwen3-8B}
host=${QWEN3_8B_HOST:-127.0.0.1}
port=${QWEN3_8B_PORT:-18080}

export MACA_PATH=${MACA_PATH:-/opt/maca}
export MACA_HOME=${MACA_HOME:-/opt/maca}
export PATH="$MACA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$python_bin" -m vllm.entrypoints.openai.api_server \
  --host "$host" \
  --port "$port" \
  --model "$model_dir" \
  --served-model-name Qwen/Qwen3-8B \
  --trust-remote-code \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --enable-prefix-caching
