import pytest
import time
from unittest.mock import MagicMock
from core.sse_relay import RedisStreamRelay, TelemetryRelay
from core.redis_client import RedisManager


@pytest.fixture
def mock_redis_mgr():
    """Fixture for mocked RedisManager."""
    mgr = MagicMock(spec=RedisManager)
    return mgr


def test_generic_relay_broadcast(mock_redis_mgr):
    """Test that RedisStreamRelay calls the broadcast callback with data from Redis."""
    raw_data = {"key": "val", "_id": "123-0"}
    mock_redis_mgr.read_stream.side_effect = [[raw_data], []]

    broadcast_mock = MagicMock()
    relay = RedisStreamRelay(mock_redis_mgr, "test:stream", broadcast_mock)

    relay.start()
    time.sleep(0.3)
    relay.stop()

    broadcast_mock.assert_called_with(raw_data)
    mock_redis_mgr.read_stream.assert_any_call("test:stream", last_id="$", block=2000)


def test_telemetry_relay_compatibility(mock_redis_mgr):
    """Verify TelemetryRelay still works as a subclass."""
    raw_data = {"total": 100, "_id": "456-0"}
    mock_redis_mgr.read_stream.side_effect = [[raw_data], []]

    broadcast_mock = MagicMock()
    relay = TelemetryRelay(mock_redis_mgr, broadcast_mock)

    relay.start()
    time.sleep(0.3)
    relay.stop()

    broadcast_mock.assert_called_with(raw_data)
    mock_redis_mgr.read_stream.assert_any_call(
        "telemetry:stream", last_id="$", block=2000
    )


def test_relay_error_resilience(mock_redis_mgr):
    """Test that relay survives errors in the broadcast callback."""
    raw_data = {"data": "stuff", "_id": "789-0"}
    mock_redis_mgr.read_stream.side_effect = [[raw_data], []]

    broadcast_mock = MagicMock(side_effect=Exception("Broadcast failed"))

    relay = RedisStreamRelay(mock_redis_mgr, "err:stream", broadcast_mock)
    relay.start()
    time.sleep(0.3)
    relay.stop()

    assert broadcast_mock.called
