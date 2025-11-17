# syntax=docker/dockerfile:1
# Dockerfile for Langflow Executor Node
# Build from repo root: docker build -f nodes/langflow-executor-node/Dockerfile -t langflow-executor-node .
# OR from droqflow directory: cd droqflow && docker build -f ../nodes/langflow-executor-node/Dockerfile -t langflow-executor-node .

################################
# BUILDER STAGE
# Build dependencies and Langflow
################################
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder

# Install build dependencies
# Retry on failure to handle transient network issues
RUN set -e; \
    for i in 1 2 3; do \
        apk update && \
        apk add --no-cache \
            build-base \
            libaio-dev \
            linux-headers && \
        break || sleep 5; \
    done

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy Langflow dependency files first (for better caching)
# These paths assume build context includes droqflow directory
COPY droqflow/app/src/lfx/pyproject.toml /app/src/lfx/pyproject.toml
COPY droqflow/app/src/lfx/README.md /app/src/lfx/README.md

# Copy executor node dependency files
COPY nodes/langflow-executor-node/pyproject.toml /app/node/pyproject.toml

# Copy Langflow source (needed for installation)
COPY droqflow/app/src/lfx/src /app/src/lfx/src

# Install Langflow (lfx) package with all dependencies
# This installs lfx and all its dependencies from pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /app/src/lfx && \
    uv pip install --system --no-cache -e .

# Install common Langchain integration packages needed by components
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache \
    "langchain-core>=0.3.79,<0.4.0" \
    langchain-anthropic \
    langchain-openai \
    langchain-community \
    langchain-google-genai \
    langchain-mistralai \
    langchain-groq \
    langchain-cohere \
    langchain-ollama \
    langchain-ibm \
    langchain-google-vertexai || echo "Warning: Some langchain packages failed to install"

# Install executor node dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    httpx \
    structlog \
    python-dotenv

# Copy executor node source
COPY nodes/langflow-executor-node/src /app/node/src

# Copy components.json mapping file
COPY nodes/langflow-executor-node/components.json /app/components.json

################################
# RUNTIME STAGE
# Minimal runtime image
################################
FROM python:3.12-alpine AS runtime

# Install runtime dependencies with retry
RUN set -e; \
    for i in 1 2 3; do \
        apk update && \
        apk add --no-cache curl && \
        break || sleep 5; \
    done

# Create non-root user
RUN adduser -D -u 1000 -G root -h /app -s /sbin/nologin executor

WORKDIR /app

# Copy Python packages from builder
# uv installs to /usr/local when using --system
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder --chown=executor:root /app/node/src /app/src
COPY --from=builder --chown=executor:root /app/src/lfx/src /app/src/lfx/src
COPY --from=builder --chown=executor:root /app/components.json /app/components.json

# Set environment variables
ENV PYTHONPATH=/app/src:/app/src/lfx/src
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LANGFLOW_EXECUTOR_NODE_URL=http://localhost:8000

# Switch to non-root user
USER executor

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the executor node
CMD ["python", "-m", "uvicorn", "node.api:app", "--host", "0.0.0.0", "--port", "8000"]
