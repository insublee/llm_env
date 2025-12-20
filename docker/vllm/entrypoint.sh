#!/bin/bash
set -e

MODEL_PATH=${MODEL_PATH:-/models/my_model}

echo "🚀 Starting vLLM server"
echo "📦 Model path: $MODEL_PATH"

vllm serve \
  "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096