#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Starting Langflow Executor Node locally..."
echo ""

# Check if uv is installed
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

# Ensure we run with the local uv-managed virtual environment
if [ -d ".venv" ]; then
    # Avoid contaminating other envs if already active
    if [ -n "$VIRTUAL_ENV" ] && [ "$VIRTUAL_ENV" != "$(pwd)/.venv" ]; then
        echo "Deactivating other virtualenv ($VIRTUAL_ENV)..."
        deactivate || true
    fi
    if [ "$VIRTUAL_ENV" != "$(pwd)/.venv" ]; then
        echo "Activating local virtualenv (.venv)..."
        # shellcheck source=/dev/null
        . ".venv/bin/activate"
    fi
fi

# Install local lfx package in editable mode so executor node uses latest code
if [ -d "lfx/src" ]; then
    echo "Installing local lfx package (editable mode)..."
    uv pip install -e ./lfx/src >/dev/null 2>&1 || true
fi

# Check if Langflow is installed (fallback to backend package if local doesn't exist)
if ! python -c "import lfx" 2>/dev/null; then
    echo "Installing Langflow dependencies from backend..."
    uv pip install -e ../app/src/lfx
fi

echo "Skipping automatic LangChain package reinstall (managed via pyproject)."

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

# Run the executor node
# Add lfx to PYTHONPATH so components can be imported directly (local source takes precedence)
export PYTHONPATH="$(pwd)/lfx/src:$(pwd)/src:${PYTHONPATH:-}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export RELOAD="${RELOAD:-true}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

# Run the lfx main module directly with Python (dependencies should already be installed)
python -c "import sys; sys.path.insert(0, 'src'); from lfx.main import main; main()"

