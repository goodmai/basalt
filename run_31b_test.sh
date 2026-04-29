#!/bin/bash
MODEL="mlx-community/gemma-4-31b-it-4bit"
echo "Downloading 31B model..."
swift run Gem models download "$MODEL"

swift run Gem serve --model "$MODEL" --rest > server_31b.log 2>&1 &
SERVER_PID=$!
echo "Waiting for 31B model to load..."
for i in {1..180}; do
  if nc -z localhost 8080; then
    echo "Server ready!"
    break
  fi
  sleep 1
done

TOKEN=$(curl -s -X POST http://localhost:8080/api/v1/auth/login -H "Content-Type: application/json" -d '{"username": "admin", "password": "admin"}' | python3 -c "import sys, json; print(json.load(sys.stdin).get('token', ''))")

echo "Running prompt..."
curl -s -X POST "http://localhost:8080/api/v1/generate" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Explain Quantum Entanglement simply.", "maxTokens": 100}' > res_31b.json

kill $SERVER_PID
cat res_31b.json
