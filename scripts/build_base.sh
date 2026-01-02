#!/bin/bash
# scripts/build_base.sh

echo "🐳 Building Training Base Image..."
docker build -t llm-train-base -f docker/train/Dockerfile.base docker/train

echo "🐳 Building vLLM Base Image..."
docker build -t llm-vllm-base -f docker/vllm/Dockerfile.base docker/vllm

echo "✅ Base images built successfully!"
