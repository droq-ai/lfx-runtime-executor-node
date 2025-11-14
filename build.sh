#!/bin/bash
# Build script for Langflow Executor Node
# Builds from repo root to access both node and app directories

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Get the repo root (parent of node directory)
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building Langflow Executor Node Docker image..."
echo "Script location: $SCRIPT_DIR"
echo "Repo root: $REPO_ROOT"
echo "Build context: $REPO_ROOT"

cd "$REPO_ROOT"

docker build \
    -f node/Dockerfile \
    -t langflow-executor-node:latest \
    .

echo ""
echo "✅ Build complete!"
echo "Run with: docker run -p 8000:8000 langflow-executor-node:latest"

