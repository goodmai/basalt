#!/bin/bash

# Load environment variables from .env
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

echo "🚀 Starting REST server with Qwen3.5-4B-4bit..."

# Start the server in the background
# We use a known port 8080
swift run Gemm serve --model mlx-community/Qwen3.5-4B-4bit --rest --port 8080 &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to be ready..."
MAX_RETRIES=30
COUNT=0
while ! curl -s http://localhost:8080/api/v1/health > /dev/null; do
  sleep 1
  COUNT=$((COUNT+1))
  if [ $COUNT -ge $MAX_RETRIES ]; then
    echo "❌ Server failed to start in time."
    kill $SERVER_PID
    exit 1
  fi
done

echo "✅ Server is ready!"

# 1. Login
echo "🔐 Logging in as $GEMM_ADMIN_USER..."
LOGIN_RESPONSE=$(curl -s -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$GEMM_ADMIN_USER\", \"password\": \"$GEMM_ADMIN_PASSWORD\"}")

TOKEN=$(echo $LOGIN_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('token', ''))")

if [ -z "$TOKEN" ]; then
  echo "❌ Login failed. Response: $LOGIN_RESPONSE"
  kill $SERVER_PID
  exit 1
fi

echo "✅ Logged in successfully. Token: ${TOKEN:0:10}..."

# 2. Generate
echo "✍️ Sending generation request..."
GENERATE_RESPONSE=$(curl -s -X POST http://localhost:8080/api/v1/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "maxTokens": 50}')

echo "📄 Response:"
echo $GENERATE_RESPONSE | python3 -m json.tool

# Stop the server
echo "🛑 Stopping server..."
kill $SERVER_PID

echo "🎉 REST test complete."
