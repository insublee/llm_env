#!/bin/bash
# scripts/run_train.sh

echo "🚀 Starting Training Service..."
docker compose up --build -d train

echo "📜 Tailing logs (Ctrl+C to stop following)..."
docker compose logs -f train
