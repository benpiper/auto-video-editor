import os
import time
import logging
import threading
import uuid
from contextlib import contextmanager
from core.redis_client import RedisManager

# Logger initialization
logger = logging.getLogger("VRAMGuard")

# Resilience & Safety Constants
DEFAULT_TIMEOUT_MS = int(os.getenv("VRAM_LOCK_TTL_MS", "60000"))
SLEEP_INTERVAL = 0.5
MAX_ACQUISITION_WAIT_MS = int(
    os.getenv("VRAM_LOCK_MAX_WAIT_MS", "300000")
)  # 5 mins default


class VRAMGuard:
    """
    Context manager that enforces a global Redis lock for VRAM intensive tasks.
    Ensures that only one worker is performing high-VRAM operations at a time.
    Uses basic SET NX PX for maximum compatibility.
    """

    LOCK_NAME = "lock:vram_atomic"

    # Lua script to release lock only if the token matches
    RELEASE_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """

    # Lua script to extend lock only if the token matches
    EXTEND_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("pexpire", KEYS[1], ARGV[2])
    else
        return 0
    end
    """

    def __init__(
        self,
        redis_manager: RedisManager,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_wait_ms: int = MAX_ACQUISITION_WAIT_MS,
    ):
        self.redis_mgr = redis_manager
        self.timeout_ms = timeout_ms
        self.max_wait_ms = max_wait_ms
        self.token = str(uuid.uuid4())
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread = None

    def _heartbeat(self):
        """Background thread to extend the lock TTL while work is in progress."""
        extend_interval = (self.timeout_ms / 1000.0) / 2.0
        while not self._stop_heartbeat.is_set():
            try:
                # Use Lua script to extend TTL
                success = self.redis_mgr.client.eval(
                    self.EXTEND_SCRIPT, 1, self.LOCK_NAME, self.token, self.timeout_ms
                )
                if success:
                    # Low log level for heartbeat to avoid noise (Medium Issue fix)
                    logger.debug(f"Extended VRAM lock TTL by {self.timeout_ms}ms")
                else:
                    logger.warning(
                        "VRAM lock heartbeat failed: Lock might have been lost or stolen."
                    )
            except Exception as e:
                logger.warning(f"Failed to extend VRAM lock heartbeat: {e}")

            # Wait for half the timeout before extending again
            self._stop_heartbeat.wait(extend_interval)

    def __enter__(self):
        """Acquire the lock and start the heartbeat."""
        logger.info(
            f"Attempting to acquire global VRAM lock (max wait: {self.max_wait_ms}ms)..."
        )
        start_time = time.time()

        while True:
            # Atomic SET NX PX
            success = self.redis_mgr.client.set(
                self.LOCK_NAME, self.token, px=self.timeout_ms, nx=True
            )

            if success:
                logger.info(f"VRAM lock acquired in {time.time() - start_time:.2f}s")
                # Start heartbeat thread to extend TTL
                self._stop_heartbeat.clear()
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat, daemon=True
                )
                self._heartbeat_thread.start()
                return self

            # High Issue fix: Prevent infinite hang
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.max_wait_ms:
                logger.error(
                    f"VRAM Lock acquisition timed out after {elapsed_ms:.0f}ms"
                )
                raise TimeoutError(
                    f"Could not acquire VRAM lock within {self.max_wait_ms}ms"
                )

            time.sleep(SLEEP_INTERVAL)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the lock and stop the heartbeat."""
        # Stop heartbeat thread
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            # We don't join for long to avoid blocking exit if thread is stuck
            self._heartbeat_thread.join(timeout=0.2)

        # Release redis lock atomically via Lua
        try:
            success = self.redis_mgr.client.eval(
                self.RELEASE_SCRIPT, 1, self.LOCK_NAME, self.token
            )
            if success:
                logger.info("VRAM lock released successfully.")
            else:
                logger.debug("VRAM lock already expired or released.")
        except Exception as e:
            logger.error(f"Error during VRAM lock release: {e}")


@contextmanager
def vram_guard(
    redis_manager: RedisManager,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    max_wait_ms: int = MAX_ACQUISITION_WAIT_MS,
):
    """Convenience functional context manager."""
    guard = VRAMGuard(redis_manager, timeout_ms, max_wait_ms)
    with guard:
        yield guard
