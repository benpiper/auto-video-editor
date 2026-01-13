import pytest
import redis
from core.redis_client import RedisManager
from fakeredis import FakeRedis
from unittest.mock import patch
from circuitbreaker import CircuitBreakerError


@pytest.fixture
def redis_mgr():
    """Fixture to provide a RedisManager associated with a FakeRedis client."""
    mgr = RedisManager(redis_url="redis://localhost:6379/0")
    # Replace the actual redis client with FakeRedis for testing
    mgr.client = FakeRedis(decode_responses=True)
    return mgr


def test_add_to_stream_success(redis_mgr):
    """Test successful XADD wrapper."""
    data = {"task": "test", "value": "123"}
    msg_id = redis_mgr.add_to_stream("test:stream", data)
    assert msg_id is not None
    # Verify via normalized read
    messages = redis_mgr.read_stream("test:stream", last_id="0")
    assert len(messages) == 1
    # Check that data is preserved and _id is added
    assert messages[0]["task"] == "test"
    assert messages[0]["_id"] == msg_id


def test_read_stream_success(redis_mgr):
    """Test successful XREAD wrapper."""
    data = {"event": "start"}
    msg_id = redis_mgr.client.xadd("test:stream", data)

    messages = redis_mgr.read_stream("test:stream", last_id="0")
    assert len(messages) == 1
    assert messages[0]["event"] == "start"
    assert messages[0]["_id"] == msg_id


def test_circuit_breaker_trips(redis_mgr):
    """Test that the circuit breaker trips after multiple failures."""
    # Mocking the client.xadd to always fail
    with patch.object(
        redis_mgr.client, "xadd", side_effect=redis.ConnectionError("Redis is down")
    ):
        # We need to call it 5 times to trip (threshold=5 defined in redis_client.py)
        for _ in range(5):
            with pytest.raises(redis.ConnectionError):
                redis_mgr.add_to_stream("fail:stream", {"msg": "broken"})

        # 6th call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            redis_mgr.add_to_stream("fail:stream", {"msg": "broken"})
