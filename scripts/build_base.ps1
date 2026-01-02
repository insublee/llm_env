# scripts/build_base.ps1

Write-Host "🐳 Building Training Base Image..."
docker build -t llm-train-base -f docker/train/Dockerfile.base docker/train

Write-Host "🐳 Building vLLM Base Image..."
docker build -t llm-vllm-base -f docker/vllm/Dockerfile.base docker/vllm

Write-Host "✅ Base images built successfully!"
