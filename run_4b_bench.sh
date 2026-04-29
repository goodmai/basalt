#!/bin/bash
MODEL="mlx-community/gemma-4-e4b-it-4bit"
echo "Downloading 4B model..."
swift run Gem models download "$MODEL"

swift run Gem serve --model "$MODEL" --rest > server_4b.log 2>&1 &
SERVER_PID=$!
echo "Waiting for 4B model to load..."
for i in {1..60}; do
  if nc -z localhost 8080; then
    echo "Server ready!"
    break
  fi
  sleep 1
done

TOKEN=$(curl -s -X POST http://localhost:8080/api/v1/auth/login -H "Content-Type: application/json" -d '{"username": "admin", "password": "admin"}' | python3 -c "import sys, json; print(json.load(sys.stdin).get('token', ''))")

echo "Running tasks for 4B..."
curl -s -X POST "http://localhost:8080/api/v1/generate" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Calculate 1234 * 5678. Just the number.", "maxTokens": 50}' > res_4b_arith.json

curl -s -X POST "http://localhost:8080/api/v1/generate" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Write a Python function to check if a number is prime.", "maxTokens": 200}' > res_4b_code.json

kill $SERVER_PID
echo "Results:"
cat res_4b_arith.json
cat res_4b_code.json
