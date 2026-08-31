#!/bin/bash
BASE_URL="http://localhost:8080"
models=(
    "mlx-community/gemma-4-26b-a4b-it-4bit"
    "mlx-community/Qwen3.6-27B-4bit"
    "mlx-community/gemma-4-31b-it-4bit"
)

# Stop any running instances
pkill -f gemm || true
sleep 2

for model in "${models[@]}"; do
    echo "--- Testing $model ---"
    
    # Try to clear memory aggressively
    pkill -f "Simulator" || true
    pkill -f "node" || true
    sudo purge || true
    
    log_file="server_${model//\//_}.log"
    .build/release/gemm serve --model "$model" --rest > "$log_file" 2>&1 &
    server_pid=$!
    
    # Wait for readiness
    for j in {1..30}; do
        ready=$(curl -s "$BASE_URL/v1/models/current" | grep -o '"is_ready":true')
        if [ "$ready" == '"is_ready":true' ]; then
            echo "Model $model is ready."
            break
        fi
        sleep 2
    done
    
    # Dump the dynamic architecture logs
    echo "Architecture Logs:"
    grep -A 6 "--- Model Architecture ---" "$log_file"
    
    echo "Running inference..."
    curl -m 15 -s "$BASE_URL/v1/chat/completions" -H "Content-Type: application/json" -d '{"model": "gemm", "messages": [{"role": "user", "content": "1+1="}], "max_tokens": 10}' > "res_${model//\//_}.json"
    
    echo "Response:"
    cat "res_${model//\//_}.json"
    echo ""
    
    kill -9 $server_pid || true
    sleep 2
done
