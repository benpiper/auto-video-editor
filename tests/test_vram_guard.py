import time
import pytest
import threading
from fakeredis import FakeRedis
from core.redis_client import RedisManager
from core.vram_guard import VRAMGuard, vram_guard


@pytest.fixture
def redis_mgr():
    """Fixture for RedisManager with FakeRedis."""
    mgr = RedisManager(redis_url="redis://localhost:6379/0")
    mgr.client = FakeRedis(decode_responses=True)
    return mgr


def test_vram_guard_acquisition_and_release(redis_mgr):
    """Verify lock is acquired and released correctly."""
    assert not redis_mgr.client.exists(VRAMGuard.LOCK_NAME)

    with vram_guard(redis_mgr):
        assert redis_mgr.client.exists(VRAMGuard.LOCK_NAME)
        # Verify the token is set
        token = redis_mgr.client.get(VRAMGuard.LOCK_NAME)
        assert token is not None

    assert not redis_mgr.client.exists(VRAMGuard.LOCK_NAME)


def test_vram_guard_contention(redis_mgr):
    """Verify that a second worker waits for the first worker to release the lock."""
    lock_acquired_event = threading.Event()
    release_lock_event = threading.Event()
    results = []

    def worker_1():
        with vram_guard(redis_mgr):
            lock_acquired_event.set()
            results.append("worker_1_locked")
            release_lock_event.wait(timeout=2.0)
        results.append("worker_1_unlocked")

    def worker_2():
        # Wait for worker_1 to get the lock
        lock_acquired_event.wait(timeout=2.0)
        with vram_guard(redis_mgr):
            results.append("worker_2_locked")
        results.append("worker_2_unlocked")

    t1 = threading.Thread(target=worker_1)
    t2 = threading.Thread(target=worker_2)

    t1.start()
    t2.start()

    # Wait for worker_1 to occupy the lock
    lock_acquired_event.wait(timeout=1.0)
    time.sleep(0.1)
    assert "worker_1_locked" in results
    assert "worker_2_locked" not in results

    # Trigger release
    release_lock_event.set()

    t1.join(timeout=3.0)
    t2.join(timeout=3.0)

    assert "worker_2_locked" in results
    assert results == [
        "worker_1_locked",
        "worker_1_unlocked",
        "worker_2_locked",
        "worker_2_unlocked",
    ]


def test_vram_guard_exception_release(redis_mgr):
    """Verify lock is released even if an exception occurs."""
    try:
        with vram_guard(redis_mgr):
            assert redis_mgr.client.exists(VRAMGuard.LOCK_NAME)
            raise ValueError("Bailing out")
    except ValueError:
        pass

    assert not redis_mgr.client.exists(VRAMGuard.LOCK_NAME)


def test_vram_guard_heartbeat_extension(redis_mgr):
    """Verify that the heartbeat extends the lock TTL."""
    # We use a short timeout for testing
    timeout_ms = 400
    with vram_guard(redis_mgr, timeout_ms=timeout_ms):
        time.sleep(0.1)
        ttl1 = redis_mgr.client.pttl(VRAMGuard.LOCK_NAME)

        time.sleep(0.3)  # Heartbeat fires around 0.2s
        ttl2 = redis_mgr.client.pttl(VRAMGuard.LOCK_NAME)

        # Second TTL should be higher or close to start if it was extended
        # because we slept 0.3s after the first check.
        # If it wasn't extended, it would be around ttl1 - 300.
        assert ttl2 > (ttl1 - 200)  # Simple check that it stays alive


def test_vram_guard_timeout(redis_mgr):
    """Verify that VRAMGuard raises TimeoutError when the lock is held elsewhere."""
    # Acquire the lock manually
    redis_mgr.client.set(VRAMGuard.LOCK_NAME, "someone_else", px=10000)

    # Attempt to acquire with a short wait
    with pytest.raises(TimeoutError):
        with vram_guard(redis_mgr, max_wait_ms=100):
            pass
