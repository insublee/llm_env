#!/bin/bash
# scripts/run_merge.sh

echo "🧬 Starting Merge Process..."
echo "   This will reuse the 'train' container to run the merge script."

docker compose run --rm train python merge_lora.py

echo "✅ Merge script finished."
