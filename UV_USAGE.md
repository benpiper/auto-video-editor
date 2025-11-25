# UV Project Setup Guide

This project is now set up with UV, a fast Python package manager.

## Quick Start

### Running the application

```bash
# Run with UV (recommended)
uv run python main.py input_video.mp4 output_video.mp4

# Or activate the virtual environment
source .venv/bin/activate
python main.py input_video.mp4 output_video.mp4
```

### Common UV Commands

```bash
# Add a new dependency
uv add package-name

# Remove a dependency
uv remove package-name

# Update dependencies
uv sync

# Run a Python script
uv run python script.py

# Run a command in the virtual environment
uv run <command>

# Create a new virtual environment
uv venv
```

## Installed Dependencies

The following packages are installed:
- **moviepy** (<2.0) - Video editing library
- **openai-whisper** - Speech recognition for filler word detection
- **pydub** - Audio processing for silence detection
- **torch** - PyTorch for Whisper model
- **numpy** - Numerical computing

## Project Structure

```
auto-video-editor/
├── .venv/              # Virtual environment (managed by UV)
├── main.py             # CLI entry point
├── processor.py        # Core video processing logic
├── create_test_video.py # Test video generator
├── pyproject.toml      # Project configuration & dependencies
├── uv.lock            # Locked dependency versions
└── README.md          # Project documentation
```

## Notes

- UV automatically manages the virtual environment in `.venv/`
- Dependencies are locked in `uv.lock` for reproducible builds
- Python 3.10+ is required (specified in `pyproject.toml`)
- UV is much faster than pip for installing packages
