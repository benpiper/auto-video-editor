import os
import pytest
import subprocess
import wave
from core.audio_extract import extract_audio_pyav


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    """Creates a 2-second synthetic video with audio."""
    tmp_dir = tmp_path_factory.mktemp("data")
    video_path = str(tmp_dir / "sample.mp4")

    # Generate a synthetic video with a sine wave audio
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=1280x720:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        video_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path


def test_extract_audio_pyav_success(sample_video, tmp_path):
    """Test successful audio extraction with specific sample rate and mono downmix."""
    output_wav = str(tmp_path / "output.wav")

    success = extract_audio_pyav(
        sample_video, output_wav, sample_rate=16000, channels=1
    )

    assert success is True
    assert os.path.exists(output_wav)

    # Verify file properties with wave module
    with wave.open(output_wav, "rb") as f:
        assert f.getframerate() == 16000
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2  # pcm_s16le is 2 bytes


def test_extract_audio_pyav_no_audio(tmp_path):
    """Test extraction from a video with no audio stream."""
    video_no_audio = str(tmp_path / "no_audio.mp4")

    # Generate video without audio
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=640x480:rate=10",
        "-c:v",
        "libx264",
        video_no_audio,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    output_wav = str(tmp_path / "failed.wav")
    success = extract_audio_pyav(video_no_audio, output_wav)

    assert success is False
    assert not os.path.exists(output_wav)


def test_extract_audio_pyav_file_not_found():
    """Test behavior when input file is missing."""
    with pytest.raises(FileNotFoundError):
        extract_audio_pyav("missing_video.mp4", "output.wav")
