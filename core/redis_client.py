import os
import redis
import logging
from typing import Optional, Dict, List, Any
from circuitbreaker import circuit

# Resilience Configuration
CB_FAILURE_THRESHOLD = 5
CB_RECOVERY_TIMEOUT = 30

# Logger initialization
logger = logging.getLogger("RedisManager")


class RedisManager:
    """
    Manages Redis connections and provides resilient stream operations.
    Includes a circuit breaker for all outbound operations.
    """

    def __init__(
        self,
        redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        socket_timeout: float = 0.2,  # 200ms threshold from AC
        socket_connect_timeout: float = 0.1,  # 100ms
        health_check_interval: int = 30,  # Fix: Avoid stale connections
    ):
        self.redis_url = redis_url
        self.is_mock = os.getenv("MOCK_REDIS", "false").lower() == "true"

        if self.is_mock:
            import fakeredis

            logger.info("Initializing RedisManager in MOCK mode (fakeredis)")
            self.client = fakeredis.FakeRedis(decode_responses=True)
            return

        try:
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                health_check_interval=health_check_interval,
                decode_responses=True,
            )
            self.client = redis.Redis(connection_pool=self.pool)

            # Quick connectivity check
            self.client.ping()

            # Security Fix: Sanitize URL for logging (mask password)
            from urllib.parse import urlparse

            parsed = urlparse(self.redis_url)
            sanitized_url = (
                f"{parsed.scheme}://{parsed.hostname}:{parsed.port}{parsed.path}"
            )
            logger.info(f"Initialized RedisManager with: {sanitized_url}")

        except (redis.ConnectionError, redis.TimeoutError) as e:
            if os.getenv("REDIS_FALLBACK_TO_MOCK", "true").lower() == "true":
                import fakeredis

                logger.warning(
                    f"Connection to Redis failed: {e}. Falling back to MOCK mode."
                )
                self.client = fakeredis.FakeRedis(decode_responses=True)
                self.is_mock = True
            else:
                logger.error(
                    f"Failed to connect to Redis and fallback is disabled: {e}"
                )
                raise

    @circuit(
        failure_threshold=CB_FAILURE_THRESHOLD, recovery_timeout=CB_RECOVERY_TIMEOUT
    )
    def add_to_stream(self, stream_name: str, data: Dict[str, Any]) -> str:
        """
        Resiliently adds data to a Redis stream (XADD).
        """
        try:
            msg_id = self.client.xadd(stream_name, data)
            return str(msg_id)
        except redis.RedisError as e:
            logger.error(f"Redis Stream Error (add_to_stream): {e}")
            raise

    @circuit(
        failure_threshold=CB_FAILURE_THRESHOLD, recovery_timeout=CB_RECOVERY_TIMEOUT
    )
    def read_stream(
        self,
        stream_name: str,
        last_id: str = "0",
        count: Optional[int] = 10,
        block: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Resiliently reads and NORMALIZES data from a Redis stream (XREAD).
        Returns a flat list of dicts with 'id' added to the payload.
        """
        try:
            streams = {stream_name: last_id}
            response = self.client.xread(streams, count=count, block=block)

            # Normalization Fix: Convert raw nested structure to flat List[Dict]
            if not response:
                return []

            messages = []
            # response format: [[stream_name, [[msg_id, data_dict], ...]]]
            for _, stream_msgs in response:
                for msg_id, data in stream_msgs:
                    data["_id"] = msg_id  # Preserve ID in the record
                    messages.append(data)
            return messages

        except redis.RedisError as e:
            logger.error(f"Redis Stream Error (read_stream): {e}")
            raise


if __name__ == "__main__":
    # Configure logging only when run as main
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    mgr = RedisManager()
    try:
        ping_ok = mgr.client.ping()
        print(f"Manager initialized: Ping OK? {ping_ok}")
    except Exception as e:
        print(f"Manager init failed (expected if local redis not running): {e}")
