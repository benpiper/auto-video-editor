import pytest
import os
from core.media_info import get_video_metadata


def test_extract_metadata_success():
    """Test successful metadata extraction from a real video."""
    video_path = "test_video.mp4"
    if not os.path.exists(video_path):
        pytest.skip("Test video missing")

    metadata = get_video_metadata(video_path)

    assert metadata["filename"] == "test_video.mp4"
    assert metadata["width"] > 0
    assert metadata["height"] > 0
    assert metadata["duration"] > 0
    assert "video" in metadata
    assert "audio" in metadata


def test_extract_metadata_not_found():
    """Test behavior when file does not exist."""
    with pytest.raises(FileNotFoundError):
        get_video_metadata("non_existent_file.mp4")


def test_extract_metadata_invalid_file(tmp_path):
    """Test behavior with an invalid (non-video) file."""
    invalid_file = tmp_path / "not_a_video.txt"
    invalid_file.write_text("This is not a video file.")

    with pytest.raises(ValueError, match="Invalid or unsupported media file"):
        get_video_metadata(str(invalid_file))
