#!/usr/bin/env python3
"""
Main entry point for Langflow Executor Node.

Runs a FastAPI server that executes Langflow components.
"""

import logging
import os
import sys

import uvicorn

try:
    from .logger import setup_logging
except ImportError:
    setup_logging = None

from .api import app

logger = logging.getLogger(__name__)


def main():
    """Main entry point - runs the FastAPI server."""
    # Setup logging
    if setup_logging:
        setup_logging()
    else:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    logger.info(f"Starting Langflow Executor Node on {host}:{port}")

    # Run the FastAPI server
    # Note: For reload to work, we need to pass app as import string
    if reload:
        uvicorn.run(
            "node.api:app",
            host=host,
            port=port,
            reload=reload,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
        )
    else:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
        )


if __name__ == "__main__":
    main()
