import os
from unittest.mock import MagicMock, patch
from processor import detect_filler_words


@patch("processor.whisper.load_model")
@patch("processor.get_available_vram_gb")
@patch("processor.torch.compile")
def test_detect_filler_words_mocked(mock_compile, mock_vram, mock_load):
    """
    Test filler word detection with a mocked Whisper model.
    """
    audio_path = "dummy.wav"
    with open(audio_path, "wb") as f:
        f.write(b"RIFF" + b"\0" * 100)

    mock_redis = MagicMock()
    mock_redis.client.set.return_value = True
    mock_redis.client.eval.return_value = 1  # for Lua release

    # Mock model object
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [
            {
                "words": [
                    {"word": " Hello", "start": 0.0, "end": 0.5},
                    {"word": " um", "start": 0.6, "end": 1.0},
                    {"word": " world", "start": 1.1, "end": 1.5},
                ]
            }
        ],
        "text": "Hello um world",
    }
    mock_load.return_value = mock_model
    mock_vram.return_value = 8.0

    intervals, transcript = detect_filler_words(
        audio_path, model_size="base", redis_mgr=mock_redis
    )

    assert len(intervals) == 1
    assert intervals[0] == (0.6, 1.0)
    assert transcript == "Hello um world"

    assert mock_redis.client.set.called

    if os.path.exists(audio_path):
        os.remove(audio_path)
