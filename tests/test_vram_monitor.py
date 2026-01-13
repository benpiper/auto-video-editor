import pytest
from unittest.mock import MagicMock, patch
import pynvml
from workers.vram_monitor import VRAMMonitor
from core.redis_client import RedisManager


@pytest.fixture
def mock_redis():
    """Fixture for mocked RedisManager."""
    return MagicMock(spec=RedisManager)


def test_vram_monitor_hardware_retrieval(mock_redis):
    """Test successful telemetry retrieval from NVML."""
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetHandleByIndex"),
        patch("pynvml.nvmlDeviceGetMemoryInfo") as mock_get_info,
    ):
        # Setup mocks
        mock_info = MagicMock()
        mock_info.total = 12 * 1024**3
        mock_info.used = 4 * 1024**3
        mock_info.free = 8 * 1024**3
        mock_get_info.return_value = mock_info

        monitor = VRAMMonitor(mock_redis)
        telemetry = monitor.get_telemetry()

        assert telemetry["total"] == mock_info.total
        assert telemetry["used"] == mock_info.used
        assert telemetry["type"] == "hardware"
        assert "timestamp" in telemetry
        assert "hostname" in telemetry


def test_vram_monitor_fallback_mode(mock_redis):
    """Test fallback to dummy data when NVML fails to init."""
    with patch(
        "pynvml.nvmlInit",
        side_effect=pynvml.NVMLError(pynvml.NVML_ERROR_LIBRARY_NOT_FOUND),
    ):
        monitor = VRAMMonitor(mock_redis)
        telemetry = monitor.get_telemetry()

        assert telemetry["type"] == "dummy"
        assert telemetry["total"] == 12 * 1024**3
        assert "timestamp" in telemetry


def test_vram_monitor_stream_push(mock_redis):
    """Test that the monitor pushes telemetry to the correct stream."""
    with patch("pynvml.nvmlInit"):
        monitor = VRAMMonitor(mock_redis)

        # Mock get_telemetry to return a fixed value
        fixed_data = {"total": 100, "used": 50, "free": 50, "type": "test"}
        monitor.get_telemetry = MagicMock(return_value=fixed_data)

        # We need to run the loop once.
        # In the new implementation, we use precise timing.
        with (
            patch("time.sleep", side_effect=KeyboardInterrupt),
            patch.object(monitor, "verify_environment"),
        ):
            monitor.run()

        # Verify XADD was called via RedisManager wrapper
        mock_redis.add_to_stream.assert_called_with("telemetry:stream", fixed_data)
