import time
import logging
import threading
from typing import Callable, Optional
from circuitbreaker import CircuitBreakerError
from core.redis_client import RedisManager

logger = logging.getLogger("RedisStreamRelay")


class RedisStreamRelay:
    """
    Relays data from a specific Redis Stream to a broadcasting function (e.g., SSE).
    """

    def __init__(
        self,
        redis_manager: RedisManager,
        stream_name: str,
        broadcast_callback: Callable[[dict], None],
        logger_name: Optional[str] = None,
    ):
        self.redis_mgr = redis_manager
        self.stream_name = stream_name
        self.broadcast_callback = broadcast_callback
        self._stop_event = threading.Event()
        self._thread = None
        self._logger = logging.getLogger(logger_name or f"Relay:{stream_name}")

    def start(self):
        """Starts the relay in a background thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._logger.info(f"Relay for {self.stream_name} started.")

    def stop(self):
        """Stops the relay thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._logger.info(f"Relay for {self.stream_name} stopped.")

    def _run(self):
        """Main relay loop."""
        last_id = "$"  # Read from now onwards

        while not self._stop_event.is_set():
            try:
                # Read from stream with blocking
                messages = self.redis_mgr.read_stream(
                    self.stream_name,
                    last_id=last_id,
                    block=2000,  # 2 seconds block
                )

                for msg in messages:
                    last_id = msg["_id"]
                    try:
                        self.broadcast_callback(msg)
                    except Exception as e:
                        self._logger.error(
                            f"Error during broadcast for {self.stream_name}: {e}"
                        )

                # Small sleep to prevent tight loop if XREAD returns instantly (though block should handle it)
                if not messages:
                    time.sleep(0.1)

            except CircuitBreakerError:
                # When circuit is open, avoid frequent retries
                if not hasattr(self, "_circuit_opened") or not self._circuit_opened:
                    self._logger.warning(
                        f"Circuit OPEN for {self.stream_name}, backing off..."
                    )
                    self._circuit_opened = True
                time.sleep(10.0)
                continue

            except Exception as e:
                self._circuit_opened = False
                error_msg = str(e)
                if not hasattr(self, "_last_error") or self._last_error != error_msg:
                    self._logger.error(f"Relay loop error for {self.stream_name}: {e}")
                    self._last_error = error_msg

                time.sleep(2.0)


# Maintain backward compatibility for now if needed, though we should update app.py
class TelemetryRelay(RedisStreamRelay):
    def __init__(
        self, redis_manager: RedisManager, broadcast_callback: Callable[[dict], None]
    ):
        super().__init__(
            redis_manager, "telemetry:stream", broadcast_callback, "TelemetryRelay"
        )
