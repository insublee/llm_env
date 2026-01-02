#!/bin/bash
# scripts/run_serve.sh


echo "🚀 Starting Serving Services (vLLM + API + WebUI)..."
docker compose up --build -d vllm api webui

echo "📜 Tailing logs (Ctrl+C to stop following)..."
docker compose logs -f vllm api webui
