#!/usr/bin/env bash

set -e

echo "=============================================="
echo "🚀 Installing GemmaServer for Apple Silicon..."
echo "=============================================="

# Check for macOS
if [ "$(uname)" != "Darwin" ]; then
    echo "❌ Error: GemmaServer requires macOS."
    exit 1
fi

# Check for Apple Silicon
if [ "$(uname -m)" != "arm64" ]; then
    echo "❌ Error: GemmaServer requires Apple Silicon (M1/M2/M3/M4)."
    exit 1
fi

# Check for Swift
if ! command -v swift &> /dev/null; then
    echo "❌ Error: Swift is not installed. Please install Xcode or Xcode Command Line Tools."
    exit 1
fi

echo "📦 Building release binary..."
swift build -c release

BIN_PATH=".build/release/GemmaServer"

if [ ! -f "$BIN_PATH" ]; then
    echo "❌ Error: Build failed, binary not found."
    exit 1
fi

echo "🔑 Requesting sudo to install to /usr/local/bin/gemma..."
sudo mkdir -p /usr/local/bin
sudo cp "$BIN_PATH" /usr/local/bin/gemma

echo "✅ Installation complete!"
echo ""
echo "Run 'gemma --help' to get started."
echo "Or jump right into a chat: 'gemma chat'"
