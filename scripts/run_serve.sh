#!/bin/bash
# scripts/run_serve.sh

echo "🚀 Starting Serving Services (vLLM + API)..."
docker compose up --build -d vllm api

echo "📜 Tailing logs (Ctrl+C to stop following)..."
docker compose logs -f vllm api
