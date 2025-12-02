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

# # Check if Langflow is installed
# if ! python -c "import lfx" 2>/dev/null; then
#     echo "Installing Langflow dependencies..."
#     uv pip install -e ../app/src/lfx
# fi

# # Install common Langchain integration packages needed by components
# # Check and install missing packages
# echo "Checking Langchain integration packages..."
# MISSING_PACKAGES=()

# if ! python -c "import langchain_anthropic" 2>/dev/null; then
#     MISSING_PACKAGES+=("langchain-anthropic")
# fi
# if ! python -c "import langchain_openai" 2>/dev/null; then
#     MISSING_PACKAGES+=("langchain-openai")
# fi
# if ! python -c "import langchain_community" 2>/dev/null; then
#     MISSING_PACKAGES+=("langchain-community")
# fi
# if ! python -c "import langchain_google_genai" 2>/dev/null; then
#     MISSING_PACKAGES+=("langchain-google-genai")
# fi
# if ! python -c "import langchain_mistralai" 2>/dev/null; then
#     MISSING_PACKAGES+=("langchain-mistralai")
# fi
# if ! python -c "import langchain_groq" 2>/dev/null; then
#     MISSING_PACKAGES+=("langchain-groq")
# fi
# if ! python -c "import langchain_cohere" 2>/dev/null; then
#     MISSING_PACKAGES+=("langchain-cohere")
# fi
# if ! python -c "import langchain_ollama" 2>/dev/null; then
#     MISSING_PACKAGES+=("langchain-ollama")
# fi
# if ! python -c "import langchain_ibm" 2>/dev/null; then
#     MISSING_PACKAGES+=("langchain-ibm")
# fi
# if ! python -c "import langchain_google_vertexai" 2>/dev/null; then
#     MISSING_PACKAGES+=("langchain-google-vertexai")
# fi

# if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
#     echo "Installing missing Langchain integration packages: ${MISSING_PACKAGES[*]}"
#     uv pip install "${MISSING_PACKAGES[@]}" || echo "Warning: Some langchain packages failed to install (this is OK if not needed)"
# else
#     echo "All common Langchain integration packages are installed."
# fi

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
# Add lfx to PYTHONPATH so components can be imported directly
export PYTHONPATH="lfx/src:src"
# Suppress HuggingFace tokenizers fork warning (transitive dep from LangChain)
export TOKENIZERS_PARALLELISM=false
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export RELOAD="${RELOAD:-true}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

# Run directly with python (dependencies should already be installed)
python -m node.main

