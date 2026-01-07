# Development Guide

This guide covers setting up the development environment, running the application, and the overall development workflow.

## 1. Prerequisites

- **Python 3.10+**
- **FFmpeg**: Required for all video processing tasks. Ensure it is accessible in your system PATH.
- **UV Power Tools**: (Recommended) This project uses [UV](https://astral-sh/uv) for dependency management.
- **NVIDIA GPU (Optional)**: Highly recommended for `whisper` transcription and background removal acceleration.

## 2. Setup

### 2.1 Dependency Installation
Using UV:
```bash
uv sync
```
Manual Installation:
```bash
pip install "moviepy<2.0" openai-whisper pydub torch numpy flask flask-cors
```

### 2.2 Environment
Create a `.env` file if you need to override default directories or ports (though currently hardcoded in `app.py`).

## 3. Local Development

### 3.1 Running the CLI
```bash
uv run python main.py <input> <output> [options]
```

### 3.2 Running the Web Portal
```bash
uv run python web_app/app.py
```
Default URL: `http://localhost:5000`

## 4. Testing
Currently, the project focuses on end-to-end manual testing.
A utility script `create_test_video.py` is provided to generate sample media with known silent periods and filler words for validation.

```bash
uv run python create_test_video.py
```

## 5. Deployment Basics

The application is currently designed for local or single-server deployment.
- **Storage**: Uploads and outputs are stored in `web_app/static/`.
- **Cleanup**: The application removes input files post-processing, but output files must be managed manually or via external scripts.
