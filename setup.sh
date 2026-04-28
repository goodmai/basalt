#!/bin/bash
# Quick setup script for GemmaServer

echo "🚀 GemmaServer Quick Setup"
echo ""

# Check if release build exists
if [ ! -f .build/release/GemmaServer ]; then
    echo "📦 Building release version..."
    swift build -c release
fi

# Option 1: System-wide install
echo ""
echo "Choose installation method:"
echo "1) Install to /usr/local/bin (requires sudo)"
echo "2) Add alias to shell config"
echo "3) Skip installation"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Installing to /usr/local/bin/gemma..."
        sudo cp .build/release/GemmaServer /usr/local/bin/gemma
        echo "✅ Done! You can now run: gemma --help"
        ;;
    2)
        SHELL_RC=""
        if [ -n "$ZSH_VERSION" ]; then
            SHELL_RC="$HOME/.zshrc"
        elif [ -n "$BASH_VERSION" ]; then
            SHELL_RC="$HOME/.bashrc"
        fi
        
        if [ -n "$SHELL_RC" ]; then
            echo "" >> "$SHELL_RC"
            echo "# GemmaServer alias" >> "$SHELL_RC"
            echo "alias gemma='swift run --package-path $(pwd) GemmaServer'" >> "$SHELL_RC"
            echo "✅ Alias added to $SHELL_RC"
            echo "Run: source $SHELL_RC"
        else
            echo "❌ Could not detect shell. Add this to your shell config:"
            echo "alias gemma='swift run --package-path $(pwd) GemmaServer'"
        fi
        ;;
    3)
        echo "Skipped. Run with: swift run GemmaServer --help"
        ;;
esac

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download a model: huggingface-cli download mlx-community/Qwen3.5-4B-4bit"
echo "2. Start server: gemma serve --model mlx-community/Qwen3.5-4B-4bit --rest"
echo "3. Test it: curl http://localhost:8080/api/v1/health"
