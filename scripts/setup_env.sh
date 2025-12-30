#!/bin/bash
# Setup script for RF-DETR development environment

set -e

echo "🚀 Setting up RF-DETR development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install development dependencies
echo "📚 Installing development dependencies..."
make install-dev

# Setup pre-commit hooks (skip if git hooks are managed by system)
echo "🪝 Setting up pre-commit hooks..."
if git config core.hooksPath >/dev/null 2>&1; then
    echo "⚠️  Git hooks are managed by your system - skipping pre-commit setup"
else
    pre-commit install
fi

# Install markdownlint globally
echo "📝 Installing markdown linting..."
if command -v npm >/dev/null 2>&1; then
    npm install -g markdownlint-cli2
    echo "✅ Markdownlint installed"
else
    echo "⚠️  npm not found - install Node.js to enable markdown linting"
fi

echo "✅ Development environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Available commands:"
echo "  make help          - Show available commands"
echo "  make lint          - Run Python linting"
echo "  make lint-fix      - Fix Python formatting"
echo "  make lint-markdown - Lint markdown files"
echo "  make test          - Run tests"
echo "  make deploy-local  - Test local inference"