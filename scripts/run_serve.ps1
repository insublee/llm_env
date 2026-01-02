# scripts/run_serve.ps1

Write-Host "🚀 Starting Serving Services (vLLM + API)..."
docker compose up --build -d vllm api

Write-Host "📜 Tailing logs (Ctrl+C to stop following)..."
docker compose logs -f vllm api
