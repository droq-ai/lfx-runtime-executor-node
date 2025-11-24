"""Basic tests for the node template."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lfx.main import main


def test_imports():
    """Test that the module can be imported."""
    from lfx import main

    assert main is not None


@pytest.mark.asyncio
async def test_main_import():
    """Test that main function can be imported and is callable."""
    assert callable(main)
    # main is a synchronous function, so we don't call it here
    # as it would start the server and block


def test_main_callable():
    """Test that main function exists and is callable."""
    assert main is not None
    assert callable(main)
