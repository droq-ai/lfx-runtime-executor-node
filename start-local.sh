#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Starting Langflow Executor Node locally..."
echo ""

# Skip uv check and dependency installation if running in Docker
if [ -n "${DOCKER_CONTAINER:-}" ]; then
    echo "Running in Docker container - skipping uv dependency installation"
    # Use the venv that was copied from the builder stage
    if [ -d "/app/.venv" ]; then
        export PATH="/app/.venv/bin:$PATH"
    fi
else
    # Local development - check if uv is installed
    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: uv is not installed. Please install it first:"
        echo "  pipx install uv"
        echo "  or visit: https://github.com/astral-sh/uv"
        exit 1
    fi

    # Install/update dependencies if needed
    if [ ! -d ".venv" ] || [ ! -f "uv.lock" ]; then
        echo "Installing dependencies..."
        uv sync
    fi
fi

# Get configuration from environment or use defaults
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-true}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Reload: $RELOAD"
echo "  Log Level: $LOG_LEVEL"
echo ""
echo "Starting executor node on http://$HOST:$PORT"
echo "Press Ctrl+C to stop"
echo ""

# Verify langchain_core.memory is available before starting
echo "Verifying langchain_core.memory availability..."
if [ -n "${DOCKER_CONTAINER:-}" ]; then
    PYTHON_CMD="/app/.venv/bin/python"
else
    PYTHON_CMD="uv run python"
fi

if ! $PYTHON_CMD -c "from langchain_core.memory import BaseMemory; print('✓ langchain_core.memory is available')" 2>/dev/null; then
    echo "ERROR: langchain_core.memory module is NOT available!"
    echo "This means langchain-core was upgraded to >=1.0.0 which removed the memory module."
    $PYTHON_CMD -c "import langchain_core; print('Current langchain-core version:', langchain_core.__version__)" 2>/dev/null || echo "langchain-core not found"
    echo ""
    echo "Please rebuild the Docker image or fix the langchain-core version."
    exit 1
fi
echo ""

# Run the executor node
# Set PYTHONPATH - preserve existing (from Dockerfile) or set for local development
if [ -z "${PYTHONPATH:-}" ]; then
    # Local development - use relative paths
    export PYTHONPATH="lfx/src:src"
fi
# Suppress HuggingFace tokenizers fork warning (transitive dep from LangChain)
export TOKENIZERS_PARALLELISM=false
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export RELOAD="${RELOAD:-true}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

# Debug: Show PYTHONPATH and current directory (helpful for troubleshooting)
echo "PYTHONPATH: ${PYTHONPATH:-not set}"
echo "Current directory: $(pwd)"

# Run directly with python (dependencies should already be installed)
# In Docker, use uvicorn directly with the app import string
if [ -n "${PYTHONPATH:-}" ] && [[ "${PYTHONPATH}" == *"/app"* ]]; then
    # Docker environment - use uvicorn directly
    cd /app
    exec python -m uvicorn node.api:app --host "${HOST}" --port "${PORT}" --log-level "${LOG_LEVEL}"
else
    # Local development - use node.main
    python -m node.main
fi
