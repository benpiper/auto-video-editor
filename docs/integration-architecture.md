# Integration Architecture

This document describes how the different parts of the Auto Video Editor project interact.

## 1. System Overview

The project consists of two primary internal boundaries:
1.  **Web Frontend (Browser)** ↔ **Flask API (Server)**
2.  **Web Portal (Server)** ↔ **Core Processor (Local Modules)**

## 2. Integration Points

### 2.1 Frontend to Backend (HTTP/SSE)
The browser interface communicates with the Flask server using standard REST patterns and Server-Sent Events.

- **File Upload**: `POST /api/upload` (Multipart/form-data).
- **Control**: `POST /api/jobs` (JSON) to initiate processing.
- **Monitoring**: `GET /progress/<job_id>` (Server-Sent Events) for real-time progress bars.

### 2.2 Web Portal to Core Processor (Python Import)
Currently, the Web Portal is tightly coupled with the Core Processor via direct function calls.

- **Entry Point**: `from processor import process_video`
- **Execution**: The web app spawns a `threading.Thread` per job to call `process_video` synchronously within that thread.
- **Reporting**: `process_video` accepts a `progress_callback` function, which updates the `Job` object in `web_app/state.py`.

### 2.3 External Dependencies
- **FFmpeg**: Invoked as a subprocess (via `moviepy` and `pydub`).
- **OpenAI Whisper**: Local model execution (PyTorch).
- **RVM / Mediapipe**: Local AI model execution.

## 3. Data Flow

1.  User uploads video → `static/uploads/`.
2.  API creates `Job` entry in `state.py`.
3.  Background thread starts → `processor.py` analyzes audio/video.
4.  Processor writes temporary fragments and merges them via FFmpeg.
5.  Final video saved to `static/outputs/`.
6.  Job status updated to `complete`.
7.  User notified via SSE/Polling → Download triggered.
