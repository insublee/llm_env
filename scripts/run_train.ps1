# scripts/run_train.ps1

Write-Host "🚀 Starting Training Service..."
docker compose up --build -d train

Write-Host "📜 Tailing logs (Ctrl+C to stop following)..."
docker compose logs -f train
