#!/bin/bash
MODEL="mlx-community/gemma-4-26b-a4b-4bit"
swift run Gem serve --model "$MODEL" --rest > server_26b.log 2>&1 &
SERVER_PID=$!
echo "Waiting for 26B model to load..."
# Heavy model, wait up to 120s
for i in {1..120}; do
  if nc -z localhost 8080; then
    echo "Server ready!"
    break
  fi
  sleep 1
done

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server crashed during load."
    cat server_26b.log
    exit 1
fi

TOKEN=$(curl -s -X POST http://localhost:8080/api/v1/auth/login -H "Content-Type: application/json" -d '{"username": "admin", "password": "admin"}' | python3 -c "import sys, json; print(json.load(sys.stdin).get('token', ''))")

echo "Running prompt..."
curl -s -X POST "http://localhost:8080/api/v1/generate" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The capital of France is", "maxTokens": 10}' > res_26b.json

kill $SERVER_PID
cat res_26b.json
