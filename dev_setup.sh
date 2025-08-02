#!/bin/bash
set -e

# Go to the project root (directory containing this script)
cd "$(dirname "$0")"

# Activate the local virtual environment
if [ -d "./venv" ]; then
  echo "Activating local virtual environment..."
  # shellcheck disable=SC1091
  source ./venv/bin/activate
else
  echo "❌ No venv found at ./venv. Please create one with 'python -m venv venv' first."
  exit 1
fi

echo "Installing local packages in editable mode..."

pip install -e ./chunking
pip install -e ./retrieval
pip install -e ./generation
pip install -e ./mcp-client
pip install -e ./keyword-extraction
pip install -e ./slack-search

echo "✅ All local packages installed in editable mode."