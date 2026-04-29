#!/bin/bash
MODEL="mlx-community/gemma-4-26b-a4b-4bit"
echo "Downloading 26B Base model..."
swift run Gem models download "$MODEL"

swift run Gem serve --model "$MODEL" --rest > server_26b_base.log 2>&1 &
SERVER_PID=$!
echo "Waiting for model to load..."
for i in {1..120}; do
  if nc -z localhost 8080; then
    echo "Server ready!"
    break
  fi
  sleep 1
done

TOKEN=$(curl -s -X POST http://localhost:8080/api/v1/auth/login -H "Content-Type: application/json" -d '{"username": "admin", "password": "admin"}' | python3 -c "import sys, json; print(json.load(sys.stdin).get('token', ''))")

echo "Running task..."
curl -s -X POST "http://localhost:8080/api/v1/generate" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The capital of France is", "maxTokens": 10}' > res_26b_base.json

kill $SERVER_PID
echo "Result:"
cat res_26b_base.json
