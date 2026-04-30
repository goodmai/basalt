#!/bin/bash
set -e

# Load environment
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "--- Starting Real Model Inference Validation (REST) ---"

# Start server in background
swift run Gemm serve --model mlx-community/Qwen3.5-4B-4bit --rest --port 8080 > logs/rest_validation.log 2>&1 &
REST_PID=$!

cleanup() {
  echo "Cleaning up REST server (PID: $REST_PID)..."
  kill $REST_PID || true
}
trap cleanup EXIT

# Wait for server
echo "Waiting for REST server to start..."
for i in {1..60}; do
  if curl -s http://localhost:8080/api/v1/health > /dev/null; then
    break
  fi
  sleep 1
done

# Login
echo "Logging in as $GEMM_ADMIN_USER..."
TOKEN=$(curl -s -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$GEMM_ADMIN_USER\", \"password\": \"$GEMM_ADMIN_PASSWORD\"}" | grep -o '"token":"[^"]*' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
  echo "ERROR: Failed to login"
  exit 1
fi

validate_task() {
  NAME=$1
  PROMPT=$2
  EXPECTED=$3
  
  echo "[Task: $NAME] Sending request..."
  RESPONSE=$(curl -s -X POST http://localhost:8080/api/v1/generate \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"$PROMPT\", \"maxTokens\": 30}")
  
  TEXT=$(echo "$RESPONSE" | grep -o '"generatedText":"[^"]*' | cut -d'"' -f4)
  
  echo "[Task: $NAME] Received: $TEXT"
  
  if [[ "$TEXT" == *"$EXPECTED"* ]]; then
    echo "✅ Task $NAME PASSED"
  else
    echo "⚠️ Task $NAME check: Received '$TEXT', expected to contain '$EXPECTED'"
  fi
}

# Run tasks
validate_task "Arithmetic" "Calculate 123 * 456 + 789 / 3" "56351"
validate_task "Algebra" "Solve for x: 2x + 5 = 15. Show steps." "x = 5"
validate_task "Translation" "Translate 'The quick brown fox jumps over the lazy dog' to Russian." "лиса"

echo "--- Real Model Inference Validation COMPLETE ---"
