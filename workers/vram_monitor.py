import time
import logging
import os
import socket
import pynvml
from workers.base import BaseWorker
from core.redis_client import RedisManager

# Logger initialization
logger = logging.getLogger("VRAMMonitor")

# Configuration Constants
POLL_INTERVAL = float(os.getenv("VRAM_POLL_INTERVAL", "2.0"))
DUMMY_TOTAL_GB = int(os.getenv("VRAM_DUMMY_TOTAL_GB", "12"))
HOSTNAME = socket.gethostname()


class VRAMMonitor(BaseWorker):
    """
    Worker that polls NVIDIA GPU telemetry and pushes it to Redis.
    Supports auto-recovery if drivers or hardware blip.
    """

    STREAM_NAME = "telemetry:stream"

    def __init__(self, redis_manager: RedisManager):
        self.redis_mgr = redis_manager
        self._initialized = False
        self._handle = None
        self._attempt_init()

    def _attempt_init(self):
        """Attempts to initialize NVML and get device handle."""
        try:
            pynvml.nvmlInit()
            # Selection of GPU 0 as per AC
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._initialized = True
            logger.info("NVML initialized successfully for GPU 0.")
        except pynvml.NVMLError as e:
            self._initialized = False
            self._handle = None
            logger.warning(
                f"NVML init failed: {e}. Worker will use dummy mode but retry hardware init later."
            )

    def get_telemetry(self):
        """Retrieves VRAM stats with auto-recovery logic."""
        # Try to recover hardware mode if we were in dummy mode
        if not self._initialized:
            self._attempt_init()

        telemetry = {
            "timestamp": time.time(),
            "hostname": HOSTNAME,
            "total": 0,
            "used": 0,
            "free": 0,
            "type": "none",
        }

        if self._initialized and self._handle:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                telemetry.update(
                    {
                        "total": info.total,
                        "used": info.used,
                        "free": info.free,
                        "type": "hardware",
                    }
                )
            except pynvml.NVMLError as e:
                logger.error(
                    f"Hardware telemetry read failed: {e}. Falling back to dummy."
                )
                self._initialized = False  # Force retry on next cycle
                self._fill_dummy_data(telemetry)
        else:
            self._fill_dummy_data(telemetry)

        return telemetry

    def _fill_dummy_data(self, telemetry: dict):
        """Fills telemetry with simulated data."""
        telemetry.update(
            {
                "total": DUMMY_TOTAL_GB * 1024**3,
                "used": 1 * 1024**3,  # Base simulate 1GB
                "free": (DUMMY_TOTAL_GB - 1) * 1024**3,
                "type": "dummy",
            }
        )

    def run(self):
        """Starts the main polling loop with drift compensation."""
        logger.info(f"Starting VRAM Monitor (Interval: {POLL_INTERVAL}s)")
        self.verify_environment()

        next_poll = time.time()

        try:
            while True:
                telemetry = self.get_telemetry()
                try:
                    self.redis_mgr.add_to_stream(self.STREAM_NAME, telemetry)
                    logger.debug(
                        f"Pushed {telemetry['type']} telemetry: {telemetry['used']}/{telemetry['total']}"
                    )
                except Exception as e:
                    logger.error(f"Failed to push telemetry to Redis: {e}")

                # Drift Compensation Fix (Medium Issue)
                next_poll += POLL_INTERVAL
                sleep_time = next_poll - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # We missed our window, reset next_poll to now
                    logger.warning("VRAM Monitor polling lag detected.")
                    next_poll = time.time()

        except KeyboardInterrupt:
            logger.info("VRAM Monitor stopping...")
        finally:
            if self._initialized:
                try:
                    pynvml.nvmlShutdown()
                    logger.info("NVML shutdown.")
                except Exception:
                    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rm = RedisManager()
    VRAMMonitor(rm).run()
