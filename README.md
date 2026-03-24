# AutoCut AI 🎬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoCut AI** is a professional-grade automated video editor that instantly cleans up your content by removing silences, filler words ("um", "uh"), and static moments. It combines OpenAI's Whisper, Robust Video Matting (RVM), and hardware-accelerated FFmpeg to transform raw recordings into polished assets.

## ✨ Key Features

- **Intelligent Silence Removal**: Automatically trims audio dead zones based on customizable amplitude thresholds.
- **Filler Word Detection**: Uses deep learning (Whisper) to identify and remove disfluencies (um, uh, etc.).
- **Visual Freeze Detection**: Identifies and removes static frames (e.g., stuck slides).
- **AI Background Removal**: Seamlessly remove backgrounds without a green screen using RVM or MediaPipe.
- **Hardware Acceleration**: Full **NVIDIA NVENC** support (H264/HEVC) for ultra-fast rendering.
- **Production Presets**: Speed Mode (hard cuts via FFmpeg) or Quality Mode (cinematic crossfades via MoviePy).
- **Real-time Progress**: Web dashboard with live processing updates.

## 🛠️ Prerequisites

- **Python 3.10+**
- **FFmpeg**: Available in your system `PATH`
- **Redis**: For real-time progress updates (optional; use `MOCK_REDIS=true` to skip)
- **NVIDIA GPU**: Optional but recommended for NVENC acceleration

## 🚀 Quick Start

### Clone & Install

```bash
git clone https://github.com/benpiper/auto-video-editor.git
cd auto-video-editor
```

This project uses [UV](https://github.com/astral-sh/uv) for dependency management. UV automatically creates a virtual environment on first run:

```bash
# Web interface (recommended)
uv run web_app/app.py
# Visit http://localhost:5000

# OR command line
uv run python main.py input.mp4 output.mp4 --min-silence 1500
```

See **[UV_USAGE.md](UV_USAGE.md)** for detailed environment setup and common commands.

### Configuration

Create a `.env` file in the root directory:

```env
REDIS_URL=redis://localhost:6379/0
MOCK_REDIS=true                          # Dev mode (skip Redis)
UPLOAD_FOLDER=web_app/static/uploads
OUTPUT_FOLDER=web_app/static/outputs
```

## 🖥️ Usage

### Web Dashboard

Visit `http://localhost:5000` to:
1. **Upload** your video
2. **Detect** silences and filler words
3. **Review** and uncheck cuts you want to keep
4. **Preview** a 30-second sandbox render
5. **Choose** Speed or Quality mode and download

See **[web_app/README.md](web_app/README.md)** for full web interface documentation and troubleshooting.

### Command Line

```bash
uv run python main.py input.mp4 output.mp4 \
    --min-silence 1500 \
    --filler-words "um;uh;like" \
    --remove-background \
    --bg-color green
```

For more examples and encoding presets, see **[QUALITY_GUIDE.md](QUALITY_GUIDE.md)**.

## 📚 Documentation

| Guide | Purpose |
|-------|---------|
| **[QUALITY_GUIDE.md](QUALITY_GUIDE.md)** | Speed vs Quality modes, hardware acceleration, encoding parameters |
| **[LOGGING_GUIDE.md](LOGGING_GUIDE.md)** | Debug detection pipeline, analyze removal decisions |
| **[UV_USAGE.md](UV_USAGE.md)** | Environment setup, dependency management, project structure |
| **[web_app/README.md](web_app/README.md)** | Web interface architecture, configuration, troubleshooting |

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
