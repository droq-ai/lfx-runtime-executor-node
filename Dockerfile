FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better layer caching
COPY pyproject.toml README.md* ./

# Install only core runtime dependencies (avoid heavy AI/ML dependencies for Docker build)
RUN pip install fastapi uvicorn pydantic httpx nats-py python-dotenv structlog

# Copy application files
COPY src/ ./src/

# Install the project (only core dependencies to avoid build issues)
RUN pip install -e . --no-deps || echo "Warning: pip install failed, but continuing"

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the application
CMD ["python", "-m", "lfx.main", "8004"]