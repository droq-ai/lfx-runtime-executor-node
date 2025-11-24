"""Tests for NATS client."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.asyncio
async def test_nats_client_import():
    """Test that NATS client can be imported."""
    try:
        from lfx.nats import NATSClient

        assert NATSClient is not None
    except ImportError:
        pytest.skip("nats-py not installed")


@pytest.mark.asyncio
async def test_nats_client_initialization():
    """Test NATS client initialization."""
    try:
        from lfx.nats import NATSClient

        client = NATSClient(nats_url="nats://localhost:4222", stream_name="test-stream")
        assert client.nats_url == "nats://localhost:4222"
        assert client.stream_name == "test-stream"
    except ImportError:
        pytest.skip("nats-py not installed")


@pytest.mark.asyncio
async def test_nats_connect(nats_url):
    """Test connecting to NATS server."""
    try:
        import uuid

        from lfx.nats import NATSClient

        # Use unique stream name to avoid conflicts
        unique_id = str(uuid.uuid4())[:8]
        stream_name = f"test-connect-{unique_id}"
        client = NATSClient(nats_url=nats_url, stream_name=stream_name)
        await client.connect()

        assert client.nc is not None
        assert client.js is not None

        await client.close()
    except ImportError:
        pytest.skip("nats-py not installed")


@pytest.mark.asyncio
async def test_nats_publish(nats_url):
    """Test publishing to NATS."""
    try:
        import uuid

        from lfx.nats import NATSClient

        # Use unique stream name to avoid conflicts
        unique_id = str(uuid.uuid4())[:8]
        stream_name = f"test-publish-{unique_id}"
        client = NATSClient(nats_url=nats_url, stream_name=stream_name)
        await client.connect()

        # Publish a test message
        await client.publish("test", {"message": "test", "value": 42})

        await client.close()
    except ImportError:
        pytest.skip("nats-py not installed")


@pytest.mark.asyncio
async def test_nats_subscribe(nats_url):
    """Test subscribing to NATS messages."""
    try:
        import asyncio
        import uuid

        from lfx.nats import NATSClient

        # Use UUID to ensure unique stream names and avoid subject conflicts
        unique_id = str(uuid.uuid4())[:8]
        stream_name = f"test-subscribe-{unique_id}"
        client = NATSClient(nats_url=nats_url, stream_name=stream_name)
        await client.connect()

        received_messages = []
        message_received = asyncio.Event()

        async def message_handler(data: dict, headers: dict):
            received_messages.append(data)
            message_received.set()

        # Use a unique subject to avoid conflicts
        subject = f"test-sub-{unique_id}"

        # Subscribe without queue (simpler for testing)
        # Note: subscribe() runs indefinitely, so we'll use a timeout
        subscribe_task = asyncio.create_task(client.subscribe(subject, message_handler))

        # Give subscription time to set up
        await asyncio.sleep(0.5)

        # Publish a message with unique content
        test_message = {"message": "hello", "test_id": f"unique-test-{unique_id}"}
        await client.publish(subject, test_message)

        # Wait for message to be received (with timeout)
        try:
            await asyncio.wait_for(message_received.wait(), timeout=2.0)
        except TimeoutError:
            pass  # Continue to check received_messages

        # Cancel subscription
        subscribe_task.cancel()
        try:
            await subscribe_task
        except asyncio.CancelledError:
            pass

        await client.close()

        # Check that message was received
        assert (
            len(received_messages) > 0
        ), f"No messages were received. Stream: {stream_name}"
        # Check that our test message is in the received messages
        test_ids = [msg.get("test_id") for msg in received_messages]
        assert (
            f"unique-test-{unique_id}" in test_ids
        ), f"Test message not found. Received: {received_messages}"
    except ImportError:
        pytest.skip("nats-py not installed")
    except asyncio.CancelledError:
        # Subscription cancellation is expected
        pass


@pytest.mark.asyncio
async def test_http_client_import():
    """Test that HTTP client can be imported."""
    try:
        from lfx.http import HTTPClient

        assert HTTPClient is not None
    except ImportError:
        pytest.skip("aiohttp not installed")


@pytest.mark.asyncio
async def test_http_client_initialization():
    """Test HTTP client initialization."""
    try:
        from lfx.http import HTTPClient

        client = HTTPClient(base_url="https://api.example.com", timeout=30)
        assert client.base_url == "https://api.example.com"
        assert client.timeout.total == 30
    except ImportError:
        pytest.skip("aiohttp not installed")
