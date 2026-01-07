# API Contracts

This document catalogs the various interfaces provided by the Auto Video Editor for integration and automation.

## 1. REST API (web_portal)

The Web Portal provides a RESTful API for managing video processing jobs. Base URL: `/api`

### 1.1 Upload File
- **Endpoint**: `POST /api/upload`
- **Description**: Upload a video file to the server for processing.
- **Request**: `multipart/form-data`
    - `file`: The video file to upload.
- **Response (200 OK)**:
    ```json
    {
      "message": "File uploaded successfully",
      "file_id": "uuid",
      "filename": "uuid_original_name.mp4",
      "original_filename": "original_name.mp4"
    }
    ```

### 1.2 Create Job
- **Endpoint**: `POST /api/jobs`
- **Description**: Start a processing job for a previously uploaded file.
- **Request**: `application/json`
    - `filename`: (Required) The filename returned by `/upload`.
    - `min_silence`: (Optional, default 2000) Minimum silence in ms.
    - `silence_thresh`: (Optional, default -63) Silence threshold in dBFS.
    - `filler_words`: (Optional, array) List of words to remove.
    - `remove_background`: (Optional, boolean) Enable RVM/Segmentation.
- **Response (201 Created)**:
    ```json
    {
      "job_id": "uuid",
      "status": "pending",
      "message": "Job started"
    }
    ```

### 1.3 Get Job Status
- **Endpoint**: `GET /api/jobs/<job_id>`
- **Description**: Check the progress and status of a job.
- **Response (200 OK)**:
    ```json
    {
      "job_id": "uuid",
      "status": "processing|complete|error",
      "progress": 45,
      "message": "Current status message",
      "download_url": "/api/jobs/uuid/download"
    }
    ```

### 1.4 Download Result
- **Endpoint**: `GET /api/jobs/<job_id>/download`
- **Description**: Download the processed video file.
- **Response**: Video file stream (`video/mp4`).

---

## 2. Command Line Interface (core)

The core processing logic can be invoked directly via `main.py`.

### 2.1 Basic Usage
```bash
python main.py input_file output_file [options]
```

### 2.2 Key Options
- `--min-silence`: Minimum silence duration in ms (default: 2000).
- `--silence-thresh`: Silence threshold in dBFS (default: -63).
- `--filler-words`: Semicolon-separated list of words.
- `--remove-background`: Enable AI background removal.
- `--use-segmentation`: Use MediaPipe segmentation instead of RVM.
- `--bitrate`: Target video bitrate (e.g., "5000k").
- `--preset`: FFmpeg encoding preset.

---

## 3. Real-time Updates (SSE)
- **Endpoint**: `GET /progress/<job_id>`
- **Description**: Server-Sent Events stream for real-time progress updates.
- **Format**: `data: {"status": "processing", "progress": 10, "message": "..."}`
