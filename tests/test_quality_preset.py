import unittest.mock as mock
from processor import get_encoding_params


@mock.patch("subprocess.run")
def test_quality_preset_nvenc_params(mocked_run):
    """Verify that 'quality' preset sets correct NVENC parameters."""
    # Mock ffmpeg -encoders to show h264_nvenc
    mocked_run.return_value.stdout = "h264_nvenc"
    mocked_run.return_value.returncode = 0

    # Test with GPU requested
    codec, preset, params = get_encoding_params(
        use_gpu=True,
        use_crf=False,
        bitrate="5000k",
        crf=18,
        preset="medium",
        render_preset="quality",
    )

    assert codec == "h264_nvenc"
    assert preset == "hq"
    # Ensure high quality params are present
    assert "-preset" in params
    assert "p7" in params
    assert "-cq" in params
    assert "18" in params


@mock.patch("subprocess.run")
def test_quality_preset_cpu_params(mocked_run):
    """Verify that 'quality' preset sets correct CPU parameters."""
    mocked_run.return_value.stdout = ""
    mocked_run.return_value.returncode = 0

    codec, preset, params = get_encoding_params(
        use_gpu=False,
        use_crf=True,
        bitrate="5000k",
        crf=18,
        preset="medium",
        render_preset="quality",
    )

    assert codec == "libx264"
    assert preset == "slower"
    assert "-crf" in params
    assert "18" in params


if __name__ == "__main__":
    test_quality_preset_nvenc_params()
    test_quality_preset_cpu_params()
    print("Quality preset tests passed logic check!")
