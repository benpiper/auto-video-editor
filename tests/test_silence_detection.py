import os
from pydub import AudioSegment
from pydub.generators import Sine
from processor import detect_silence


def test_detect_silence_synthetic(tmp_path):
    """
    Test silence detection with a synthetic audio file containing specific gaps.
    """
    audio_path = str(tmp_path / "silence_test.wav")

    # Create 2 seconds of sound, 2 seconds of silence, 2 seconds of sound
    sound = Sine(1000).to_audio_segment(duration=2000)
    silence_seg = AudioSegment.silent(duration=2000)

    combined = sound + silence_seg + sound
    combined.export(audio_path, format="wav")

    # Progress callback mock
    progress_calls = []

    def progress_cb(p, m):
        progress_calls.append(p)

    intervals = detect_silence(
        audio_path,
        min_silence_len=500,
        silence_thresh=-50,
        progress_callback=progress_cb,
    )

    # Should find one interval around (2.0, 4.0)
    assert len(intervals) == 1
    start, end = intervals[0]
    assert 1.9 <= start <= 2.1
    assert 3.9 <= end <= 4.1

    # Verify progress was reported
    assert len(progress_calls) > 0


def test_detect_silence_no_silence(tmp_path):
    """Test with continuous sound."""
    audio_path = str(tmp_path / "noisy.wav")
    sound = Sine(1000).to_audio_segment(duration=3000)
    sound.export(audio_path, format="wav")

    intervals = detect_silence(audio_path, min_silence_len=500, silence_thresh=-50)
    assert len(intervals) == 0
