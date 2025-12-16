# Auto Video Editor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python tool that automatically detects and removes silence, filler words (like "um" and "uh"), and static moments from video files. Perfect for cleaning up screen recordings, lectures, podcasts, and presentations.

## Features

- **Silence Detection**: Automatically removes silent segments based on audio amplitude
- **Filler Word Removal**: Uses OpenAI's Whisper model to transcribe and identify filler words (customizable list)
- **Freeze Frame Detection**: Automatically removes moments where the screen is static for too long
- **Background Removal**: Remove video backgrounds using AI-powered Robust Video Matting (RVM) - no green screen required!
- **High-Speed Processing**: Uses FFmpeg for fast video concatenation and stream copying where possible
- **High-Quality Output**: Configurable bitrate and encoding settings to maintain video quality
- **Web Interface**: User-friendly web UI for easy uploading and configuration
- **GPU Acceleration**: Leverages NVIDIA CUDA for fast Whisper transcription and RVM processing

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
pip install "moviepy<2.0" openai-whisper pydub torch numpy flask
```

*Note: Python 3.10+ is required.*

## Usage

### Web Interface (Recommended)

Start the web server:
```bash
uv run web_app/app.py
```
Then open `http://localhost:5000` in your browser.

### REST API

The application provides a full REST API for automation.

**Documentation**: `http://localhost:5000/api/docs` (Swagger UI)

**Endpoints**:
- `POST /api/upload`: Upload video file (returns `filename` for next step)
- `POST /api/jobs`: Start processing (accepts JSON params)
- `GET /api/jobs/<job_id>`: Check status/progress
- `GET /api/jobs/<job_id>/download`: Download result (optional `?delete_after=true`)

**Example Workflow**:
```bash
# 1. Upload
curl -X POST -F "file=@input.mp4" http://localhost:5000/api/upload

# 2. Process (using filename from step 1)
curl -X POST http://localhost:5000/api/jobs \
     -H "Content-Type: application/json" \
     -d '{"filename": "uuid_input.mp4", "min_silence": 1000}'

# 3. Download (when status is "complete")
curl -O -J http://localhost:5000/api/jobs/<job_id>/download
```

### Command Line

```bash
# Basic usage
uv run python main.py input.mp4 output.mp4

# With custom settings
uv run python main.py input.mp4 output.mp4 \
    --min-silence 1500 \
    --silence-thresh -63 \
    --filler-words "um;uh;like" \
    --freeze-duration 5
```

### Command-Line Options

#### Detection Parameters
- `--min-silence` - Minimum silence duration in milliseconds (default: `2000`)
- `--silence-thresh` - Silence threshold in dBFS (default: `-40`)
- `--filler-words` - Semicolon-separated list of filler words (default: `um;uh;umm;uhh;er;just;you know;like;you know`)
- `--freeze-duration` - Remove still moments longer than this (seconds). Disabled by default.
- `--freeze-noise` - Noise tolerance for freeze detection (default: `0.001`)
- `--remove-background` - Enable background removal using RVM
- `--bg-color` - Background color or 'transparent' (default: `green`)
- `--bg-image` - Path to background image file (takes precedence over `--bg-color`)
- `--rvm-model` - RVM model variant: `mobilenetv3` (faster) or `resnet50` (higher quality)
- `--rvm-downsample` - Downsample ratio for RVM (default: auto-detect based on resolution)
- `--use-segmentation` - Use person segmentation instead of RVM (better for handling occlusions)
- `--seg-model` - Segmentation model: `general` (default) or `landscape` (for portrait videos)
- `--seg-threshold` - Confidence threshold for person detection (0.0-1.0, default: `0.5`)
- `--seg-smooth` - Mask smoothing radius in pixels (default: `5`, 0 to disable)


#### Video Quality Parameters
- `--bitrate` - Target video bitrate (default: `5000k`)
- `--crf` - CRF quality value (default: `18`, used with `--use-crf`)
- `--preset` - Encoding speed preset (default: `medium`)
- `--no-crossfade` - Disable crossfades for faster processing (default: `False`)

### Examples

**Clean up a lecture:**
```bash
uv run python main.py lecture.mp4 lecture-edited.mp4 \
    --min-silence 1500 \
    --filler-words "um;uh;so;basically" \
    --freeze-duration 10
```

**Fast processing for preview:**
```bash
uv run python main.py video.mp4 preview.mp4 \
    --preset veryfast --bitrate 2000k --no-crossfade
```

**Remove background from a video:**
```bash
uv run python main.py input.mp4 output.mp4 --remove-background
```

**Remove background with custom color:**
```bash
uv run python main.py input.mp4 output.mp4 \
    --remove-background --bg-color "#FFFFFF"
```

**Remove background with transparency (for compositing):**
```bash
uv run python main.py input.mp4 output.mov \
    --remove-background --bg-color transparent
```

**Replace background with custom image:**
```bash
uv run python main.py input.mp4 output.mp4 \
    --remove-background --bg-image path/to/background.jpg
```

**Use person segmentation (better for occlusions):**
```bash
uv run python main.py input.mp4 output.mp4 --use-segmentation
```

**Segmentation with custom background:**
```bash
uv run python main.py input.mp4 output.mp4 \
    --use-segmentation --bg-color "#0000FF"
```

**Segmentation for portrait videos:**
```bash
uv run python main.py selfie.mp4 output.mp4 \
    --use-segmentation --seg-model landscape
```

**Full processing with all features:**
```bash
uv run python main.py lecture.mp4 final.mp4 \
    --min-silence 1500 \
    --filler-words "um;uh;like" \
    --freeze-duration 5 \
    --use-segmentation --bg-color green
```

## How It Works

1. **Audio Extraction**: Extracts audio track (handles silent videos gracefully)
2. **Analysis**:
   - **Silence**: Detects silent intervals using `pydub`
   - **Filler Words**: Transcribes with `whisper` to find specific words
   - **Freeze Frames**: Uses FFmpeg to detect static visual scenes
3. **Merging**: Combines all removal intervals to optimize cuts
4. **Processing**: Uses FFmpeg to extract and concatenate segments efficiently
5. **Background Removal** (optional): 
   - **RVM Matting**: AI-powered matting for single subjects
   - **Person Segmentation**: MediaPipe-based segmentation (better for occlusions, multiple people)
6. **Encoding**: Applies high-quality encoding settings


## Documentation

- **[UV_USAGE.md](UV_USAGE.md)** - Guide to using UV package manager
- **[QUALITY_GUIDE.md](QUALITY_GUIDE.md)** - Detailed explanation of quality settings
- **[LOGGING_GUIDE.md](LOGGING_GUIDE.md)** - How to use and interpret the detailed logs

## Requirements

- Python 3.10+
- FFmpeg (must be installed on system)
- NVIDIA GPU with CUDA support (optional, recommended for Whisper speed)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

