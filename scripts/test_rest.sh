#!/bin/bash

# Load environment variables from .env
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

echo "🚀 Starting REST server with Qwen3.5-4B-4bit..."

# Start the server in the background
# We use a known port 8080
swift run Basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest --port 8080 &
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

# No login route: the server checks a static API key. On loopback none is needed unless one
# is configured — BASALT_API_KEY (the server reads it too) or `api_key` in ~/.gem/config.json;
# in the latter case export the same value as BASALT_API_KEY before running this script.
TOKEN="${BASALT_API_KEY:-}"

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
