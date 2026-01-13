# UV Project Setup Guide 🛠️

AutoCut AI uses **UV** for high-performance dependency management and reproducible environments.

## 🚀 Quick Start

### Running the application

The most reliable way to run AutoCut AI is using `uv run`. This ensures all dependencies are correctly loaded in an isolated environment.

```bash
# Start the Web App
uv run web_app/app.py

# Run the CLI
uv run python main.py input_video.mp4 output_video.mp4
```

*Note: The first time you run a command, UV will automatically create a `.venv` and install all required packages.*

### Common UV Commands

```bash
# Add a new dependency
uv add package-name

# Sync environment with pyproject.toml
uv sync

# Run a Python script
uv run python script.py

# Access the virtual environment directly
source .venv/bin/activate
```

## 📦 Core Dependencies

AutoCut AI relies on the following heavy-duty libraries:
- **flask** & **flask-sse**: Web server and real-time event broadcasting.
- **peewee**: SQLite ORM for project and cut management.
- **redis**: Powering the SSE stream and progress telemetry.
- **openai-whisper**: AI transcription for filler word detection.
- **moviepy**: High-fidelity clip concatenation and crossfades.
- **ffmpegcv**: Hardware-accelerated frame reading and writing.
- **torch**: Deep learning backend for Whisper and RVM.
- **pydub**: Waveform analysis for silence detection.

## 📁 Project Structure

```
auto-video-editor/
├── .venv/              # Isolation layer (managed by UV)
├── main.py             # Scriptable CLI entry point
├── processor.py        # Core processing & AI logic
├── web_app/            # Flask-based portal & dashboard
├── core/               # Shared utilities (DB, Redis, Telemetry)
├── pyproject.toml      # Manifest & Dependency constraints
└── uv.lock            # Deterministic lockfile
```

## 💡 Notes

- **Version Lock**: Always use `uv sync` if you see errors after a fresh git pull.
- **Python**: 3.10+ is required as defined in `pyproject.toml`.
- **Speed**: UV is significantly faster than `pip` and handles complex torch/cuda versioning better.
