import os
import pytest
import subprocess
from processor import extract_sandbox_segment


def test_extract_sandbox_segment_success(tmp_path):
    # Create a dummy video file using ffmpeg
    input_video = str(tmp_path / "input.mp4")
    output_video = str(tmp_path / "sandbox.mp4")

    # Generate 5 seconds of black video
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x480:d=5",
            "-c:v",
            "libx264",
            "-t",
            "5",
            input_video,
        ],
        check=True,
        capture_output=True,
    )

    # Extract 2 seconds
    success = extract_sandbox_segment(input_video, output_video, duration=2.0)

    assert success is True
    assert os.path.exists(output_video)

    # Verify duration of output
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        output_video,
    ]
    duration = float(subprocess.check_output(ffprobe_cmd).decode().strip())
    # FFMpeg copy might not be exact due to keyframes, but for 5s it should be close
    assert duration >= 1.9


def test_extract_sandbox_segment_file_not_found():
    success = extract_sandbox_segment("non_existent_file.mp4", "output.mp4")
    assert success is False
