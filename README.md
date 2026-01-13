# AutoCut AI 🎬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoCut AI** is a professional-grade automated video editor designed to instantly clean up your content by removing silences, filler words ("um", "uh"), and stale moments. It combines the power of OpenAI's Whisper, Robust Video Matting (RVM), and hardware-accelerated FFmpeg to transform raw recordings into polished assets.

## ✨ Key Features

- **Intelligent Silence Removal**: Automatically trims audio dead zones based on customizable amplitude thresholds.
- **Filler Word Detection**: Uses deep learning (Whisper) to identify and remove disfluencies (um, uh, error, etc.).
- **Visual Freeze Detection**: Identifies static frames (e.g., slides stuck for too long) and cleans them up.
- **AI Background Removal**: Seamlessly remove video backgrounds without a green screen using RVM or MediaPipe Segmentation.
- **Hardware Acceleration**: Full support for **NVIDIA NVENC** (H264/HEVC) for ultra-fast rendering on supported GPUs.
- **Production Presets**:
  - 🚀 **Speed Mode**: Near-instant hard cuts using FFmpeg stream copying.
  - ✨ **Quality Mode**: Cinematic crossfades and audio smoothing using MoviePy.
- **Real-time Telemetry**: Monitoring of VRAM, GPU usage, and rendering progress via a modern web dashboard.
- **Web Portal**: Easy-to-use interface with a "30s Sandbox" to preview settings before the final master render.

## 🛠️ Prerequisites

Before running AutoCut AI, ensure your system meets these requirements:

1.  **Python 3.10+**
2.  **FFmpeg**: Must be installed and available in your system `PATH`.
3.  **Redis**: Required for real-time progress updates and SSE telemetry.
4.  **NVIDIA GPU (Optional but Recommended)**: Required for NVENC accelerated rendering and faster Whisper inference.

## 🚀 Installation

This project uses [UV](https://github.com/astral-sh/uv) for high-performance dependency management.

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/benpiper/auto-video-editor.git
    cd auto-video-editor
    ```

2.  **Install Dependencies**:
    UV will automatically create a virtual environment and load dependencies when you run any command:
    ```bash
    uv run web_app/app.py
    ```

    *Alternatively, for traditional pip:*
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

AutoCut AI can be configured via environment variables. Create a `.env` file in the root directory:

| Variable        | Description                                             | Default                    |
| :-------------- | :------------------------------------------------------ | :------------------------- |
| `REDIS_URL`     | Connection string for the Redis server.                 | `redis://localhost:6379/0` |
| `DB_PATH`       | Path to the SQLite database file for projects.          | `projects.db`              |
| `MOCK_REDIS`    | Set to `true` to run without a Redis server (Dev only). | `false`                    |
| `UPLOAD_FOLDER` | Directory for temporary video uploads.                  | `web_app/static/uploads`   |
| `OUTPUT_FOLDER` | Directory where final renders are stored.               | `web_app/static/outputs`   |

## 🖥️ Usage

### Web Dashboard (Recommended)

Start the web application:
```bash
uv run web_app/app.py
```
Visit `http://localhost:5000` to:
1.  **Ingest**: Upload your raw video file.
2.  **Detect**: Run the AI detection pipeline (Silences/Fillers).
3.  **Review**: Uncheck any cuts you want to keep.
4.  **Sandbox**: Generate a 30s preview to verify transitions and background removal.
5.  **Render**: Choose **Speed** or **Quality** mode and download your master.

### Command Line Interface

```bash
uv run python main.py input.mp4 output.mp4 \
    --min-silence 1500 \
    --filler-words "um;uh;like" \
    --remove-background \
    --bg-color green
```

## 📖 Advanced Documentation

- **[QUALITY_GUIDE.md](QUALITY_GUIDE.md)** - Deep dive into H264 vs HEVC and Preset logic.
- **[LOGGING_GUIDE.md](LOGGING_GUIDE.md)** - How to debug the detection pipeline.
- **[UV_USAGE.md](UV_USAGE.md)** - Tips for managing the Python environment with UV.

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
