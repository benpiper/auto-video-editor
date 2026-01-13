import os
import pytest
import subprocess
from processor import extract_sandbox_segment, assemble_preview_video


def test_assemble_preview_video(tmp_path):
    # 1. Create a dummy video (5 seconds)
    input_video = str(tmp_path / "input.mp4")
    sandbox_video = str(tmp_path / "sandbox.mp4")
    preview_video = str(tmp_path / "preview.mp4")

    # Generate 5s of color video with some visible difference (e.g. counter)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=5:size=640x480:rate=30",
            "-c:v",
            "libx264",
            input_video,
        ],
        check=True,
        capture_output=True,
    )

    # 2. Extract sandbox (first 3 seconds)
    extract_sandbox_segment(input_video, sandbox_video, duration=3.0)

    # 3. Define cuts (remove 1.0s to 2.0s)
    # This should leave 0-1s and 2-3s (total 2s)
    removal_intervals = [(1.0, 2.0)]

    success = assemble_preview_video(
        sandbox_video, preview_video, removal_intervals, max_duration=3.0
    )

    assert success is True
    assert os.path.exists(preview_video)

    # 4. Verify duration of preview (~2s)
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        preview_video,
    ]
    duration = float(subprocess.check_output(ffprobe_cmd).decode().strip())
    # Precision might vary slightly due to frame boundaries
    assert 1.9 <= duration <= 2.1
