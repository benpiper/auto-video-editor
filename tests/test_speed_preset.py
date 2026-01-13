import unittest.mock as mock
from processor import get_encoding_params


def test_speed_preset_nvenc_params():
    """Verify that 'speed' preset sets correct NVENC parameters."""
    with mock.patch("subprocess.run") as mocked_run:
        mocked_run.return_value.stdout = "h264_nvenc"
        mocked_run.return_value.returncode = 0

        # Test with GPU requested
        codec, preset, params = get_encoding_params(
            use_gpu=True,
            use_crf=False,
            bitrate="8000k",
            crf=18,
            preset="medium",
            render_preset="speed",
        )

        assert codec == "h264_nvenc"
        assert preset == "hp"
        assert "-preset" in params
        assert "p1" in params
        assert "-tune" in params
        assert "ll" in params


def test_quality_preset_nvenc_params():
    """Verify that other presets use standard mapping."""
    with mock.patch("subprocess.run") as mocked_run:
        mocked_run.return_value.stdout = "h264_nvenc"
        mocked_run.return_value.returncode = 0

        codec, preset, params = get_encoding_params(
            use_gpu=True,
            use_crf=False,
            bitrate="5000k",
            crf=18,
            preset="slow",
            render_preset="quality",
        )

        assert codec == "h264_nvenc"
        assert preset == "hq"
        assert params == []  # Default params for non-speed mode


if __name__ == "__main__":
    test_speed_preset_nvenc_params()
    test_quality_preset_nvenc_params()
    print("Tests passed logic check!")
