#!/bin/bash
TOKEN=$(curl -s -X POST http://localhost:8080/api/v1/auth/login -H "Content-Type: application/json" -d '{"username": "admin", "password": "admin"}' | python3 -c "import sys, json; print(json.load(sys.stdin).get('token', ''))")

curl -s -X POST "http://localhost:8080/api/v1/generate" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The capital of France is Paris. The capital of Germany is", "maxTokens": 5}' > res_26b_base.json

cat res_26b_base.json
