# Auto Video Editor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python tool that automatically detects and removes silence and filler words (like "um" and "uh") from video files, applying smooth crossfade transitions between cuts. Perfect for cleaning up screen recordings, lectures, podcasts, and presentations.

## Features

- **Silence Detection**: Automatically removes silent segments based on audio amplitude
- **Filler Word Removal**: Uses OpenAI's Whisper model with GPU acceleration to transcribe and identify filler words
- **Smooth Transitions**: Applies crossfades to audio and video to avoid jarring jump cuts
- **High-Quality Output**: Configurable bitrate and encoding settings to maintain video quality
- **Detailed Logging**: See exactly what's being removed with timestamp-level precision
- **GPU Acceleration**: Leverages NVIDIA CUDA for fast Whisper transcription

## Installation

This project uses [UV](https://github.com/astral-sh/uv) for fast, reliable dependency management.

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd auto-video-editor
   ```

2. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Dependencies are already configured** - UV will automatically install them when you run the tool!

### Alternative: Traditional pip Installation

If you prefer using pip:
```bash
pip install "moviepy<2.0" openai-whisper pydub torch numpy
```

*Note: Python 3.10+ is required.*

## Usage

### Basic Usage

```bash
# Using UV (recommended)
uv run python main.py input.mp4 output.mp4

# Or with traditional Python
python main.py input.mp4 output.mp4
```

### Recommended Settings

```bash
# For screen recordings and lectures
uv run python main.py my_video.mp4 edited_video.mp4 \
    --min-silence 1500 \
    --silence-thresh -63 \
    --crossfade 0.2 \
    --bitrate 5000k

# For podcasts and interviews
uv run python main.py podcast.mp4 edited_podcast.mp4 \
    --min-silence 1000 \
    --silence-thresh -63 \
    --bitrate 3000k

# Quick processing (lower quality, faster)
uv run python main.py video.mp4 output.mp4 \
    --preset veryfast \
    --bitrate 2000k
```

### Command-Line Options

#### Detection Parameters
- `--min-silence` - Minimum silence duration in milliseconds (default: `2000`)
  - Lower values = more aggressive silence removal
  - Recommended: `1000-2000` for most videos
- `--silence-thresh` - Silence threshold in dBFS (default: `-40`)
  - Higher values (e.g., `-30`) = more aggressive
  - Lower values (e.g., `-50`) = less aggressive
  - Recommended: `-60` to `-70` for most videos

#### Video Quality Parameters
- `--bitrate` - Target video bitrate (default: `5000k`)
  - Higher = better quality, larger files
  - Examples: `760k`, `2000k`, `5000k`, `10000k`
- `--crf` - CRF quality value, 0-51 (default: `18`, only used with `--use-crf`)
  - Lower = better quality
  - `18` = visually lossless, `23` = good quality
- `--preset` - Encoding speed preset (default: `medium`)
  - Options: `ultrafast`, `veryfast`, `fast`, `medium`, `slow`, `slower`, `veryslow`
  - Slower = better compression, longer encoding time
- `--use-crf` - Use CRF mode instead of constant bitrate (optional)
  - Better for videos with varying complexity
  - May produce lower bitrate for simple content

#### Other Options
- `--crossfade` - Crossfade duration in seconds (default: `0.1`)
  - Recommended: `0.1-0.3` for smooth transitions

### Examples

**High-quality screen recording:**
```bash
uv run python main.py lecture.mp4 lecture-edited.mp4 \
    --min-silence 1500 --silence-thresh -63 \
    --bitrate 5000k --crossfade 0.2
```

**Fast processing for quick preview:**
```bash
uv run python main.py video.mp4 preview.mp4 \
    --preset veryfast --bitrate 2000k
```

**Maximum quality (slow):**
```bash
uv run python main.py video.mp4 output.mp4 \
    --use-crf --crf 15 --preset slow
```

## How It Works

1. **Audio Extraction**: The audio track is extracted from the video file
2. **Silence Detection**: `pydub` analyzes the audio to find silent intervals
3. **Filler Word Detection**: `whisper` (with GPU acceleration) transcribes the audio and identifies filler words with precise timestamps
4. **Interval Merging**: Overlapping removal intervals are merged to optimize cuts
5. **Video Editing**: The video is cut to keep only desired segments, with crossfade transitions applied using `moviepy`
6. **High-Quality Encoding**: Output is encoded with configurable quality settings

## GPU Support

This tool automatically uses NVIDIA GPU acceleration if available:
- **Whisper transcription** runs on GPU for 5-10x faster processing
- Automatically detects CUDA and uses it when present
- Falls back to CPU if no GPU is available

To verify GPU usage:
```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Detailed Logging

The tool provides detailed logs showing:
- Each silence interval detected with timestamps
- Each filler word found with precise timing
- Merged removal segments
- Summary statistics (original duration, final duration, time saved)

Example output:
```
INFO - Found 15 silence intervals:
INFO -   Silence 1: 12.50s - 15.30s (duration: 2.80s)
INFO -   Filler word detected: 'um' at 5.23s - 5.45s (duration: 0.22s)
INFO - Total duration to be removed: 67.45s
INFO - Original video duration: 300.00s
INFO - Final video duration: 232.55s (removed 67.45s)
```

See [LOGGING_GUIDE.md](LOGGING_GUIDE.md) for more details.

## Documentation

- **[UV_USAGE.md](UV_USAGE.md)** - Guide to using UV package manager
- **[QUALITY_GUIDE.md](QUALITY_GUIDE.md)** - Detailed explanation of quality settings
- **[BITRATE_FIX.md](BITRATE_FIX.md)** - Understanding bitrate modes (CRF vs constant)
- **[LOGGING_GUIDE.md](LOGGING_GUIDE.md)** - How to use and interpret the detailed logs

## Troubleshooting

### Low output bitrate
By default, the tool uses **constant bitrate mode** to ensure predictable quality. If you're getting low bitrate output, make sure you're using the latest version and specify `--bitrate` explicitly.

### GPU not detected
Ensure you have CUDA-compatible PyTorch installed. The UV setup includes CUDA support by default.

### Filler words not detected
- Whisper may not catch all filler words (it's trained to remove them)
- Try adjusting silence detection parameters instead
- Check the logs to see what Whisper is transcribing

## Requirements

- Python 3.10+
- FFmpeg (usually bundled with moviepy)
- NVIDIA GPU with CUDA support (optional, but recommended for speed)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [MoviePy](https://github.com/Zulko/moviepy) for video editing
- [PyDub](https://github.com/jiaaro/pydub) for audio processing
