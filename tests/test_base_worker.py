import pytest
import collections
from unittest.mock import patch
from workers.base import BaseWorker

usage = collections.namedtuple("usage", "total used free")


def test_verify_environment_fails_on_missing_path():
    """Test that BaseWorker.verify_environment raises SystemExit when /dev/shm does not exist."""
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False

        with pytest.raises(SystemExit) as excinfo:
            BaseWorker.verify_environment()

        assert excinfo.value.code == 1


def test_verify_environment_fails_on_low_space():
    """Test that BaseWorker.verify_environment raises SystemExit when /dev/shm space is low."""
    with (
        patch("os.path.exists") as mock_exists,
        patch("shutil.disk_usage") as mock_usage,
    ):
        mock_exists.return_value = True
        mock_usage.return_value = usage(
            total=10 * 1024**3, used=9 * 1024**3, free=1 * 1024**3
        )

        with pytest.raises(SystemExit) as excinfo:
            BaseWorker.verify_environment()

        assert excinfo.value.code == 1


def test_verify_environment_passes_on_sufficient_space():
    """Test that BaseWorker.verify_environment does not raise SystemExit when /dev/shm space is sufficient."""
    with (
        patch("os.path.exists") as mock_exists,
        patch("shutil.disk_usage") as mock_usage,
    ):
        mock_exists.return_value = True
        mock_usage.return_value = usage(
            total=10 * 1024**3, used=7 * 1024**3, free=3 * 1024**3
        )

        # Should not raise any exception
        BaseWorker.verify_environment()
