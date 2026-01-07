# Source Tree Analysis

This document provides an annotated overview of the project structure for the Auto Video Editor.

## Directory Structure

```text
auto_video_editor/
├── main.py                 # Primary CLI Entry Point
├── processor.py            # Core logic: orchestrates analysis and editing
├── background_remover.py   # RVM-based background removal logic
├── person_segmenter.py     # MediaPipe-based person segmentation logic
├── pyproject.toml          # Project metadata and dependencies (managed via UV)
├── runner.py               # (Likely) Auxiliary script for running tasks
├── web_app/                # Web Dashboard (Flask Application)
│   ├── app.py              # Flask server and main web entry point
│   ├── api.py              # REST API blueprint definitions
│   ├── state.py            # In-memory job state monitoring
│   ├── static/             # Static frontend assets
│   │   ├── css/            # Frontend styling
│   │   ├── js/             # Application logic (AJAX/SSE)
│   │   └── swagger.json    # API specification
│   └── templates/          # Jinja2 HTML templates
├── docs/                   # System-generated documentation
└── _bmad*                  # BMAD internal data and outputs
```

## Critical Files & Folders

### 🎯 Entry Points
- `main.py`: The main script for command-line users. Handles argument parsing and calls the processing pipeline.
- `web_app/app.py`: Starts the Flask development server for the web interface.

### ⚙️ Core Logic
- `processor.py`: The "brain" of the application. It manages the flow between audio analysis (silence/filler detection), video scanning (freeze frames), and final editing (FFmpeg concatenation).
- `background_remover.py` & `person_segmenter.py`: Specialist modules for AI-powered visual tasks.

### 🌐 Web Portal
- `web_app/api.py`: Decouples the REST API from the main web server logic.
- `web_app/state.py`: Critical for multi-user job tracking as it manages the global `jobs` dictionary and progress updates.

### 📋 Documentation & Guides
- `README.md`: The primary user manual.
- `QUALITY_GUIDE.md`: Essential for understanding the FFmpeg/MoviePy encoding parameters.
- `LOGGING_GUIDE.md`: Describes how to troubleshoot processing failures using `run.log`.
