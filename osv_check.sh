#!/bin/bash
DEPS=("mlx-swift" "mlx-swift-lm" "hummingbird" "swift-argument-parser" "swift-transformers" "sqlite.swift" "jwt-kit" "swift-crypto")
for dep in "${DEPS[@]}"; do
    echo "Checking $dep..."
    curl -s "https://api.osv.dev/v1/query" -d "{\"package\": {\"name\": \"$dep\", \"ecosystem\": \"SwiftURL\"}}" | jq .
done
