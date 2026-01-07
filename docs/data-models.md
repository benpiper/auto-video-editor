# Data Models

This document describes the data structures used within the Auto Video Editor system.

## 1. Internal State Models

### 1.1 Job Object (web_app)
Tracks the state of a processing request in the web portal and API.

| Field         | Type          | Description                                              |
| :------------ | :------------ | :------------------------------------------------------- |
| `job_id`      | String (UUID) | Unique identifier for the job.                           |
| `filename`    | String        | Original filename or temporary handle.                   |
| `status`      | Enum          | `pending`, `processing`, `complete`, `error`, `skipped`. |
| `progress`    | Integer       | Percentage complete (0-100).                             |
| `message`     | String        | Human-readable status update.                            |
| `created_at`  | DateTime      | Timestamp when the job was created.                      |
| `output_path` | String        | Path to the final processed video file.                  |
| `error`       | String        | Error message if status is `error`.                      |
| `transcript`  | String        | Full text transcription (if applicable).                 |

## 2. Parameter Models

### 2.1 Processing Parameters
Parameters passed from the UI/API to the core processor.

| Parameter        | Type    | Default | Description                     |
| :--------------- | :------ | :------ | :------------------------------ |
| `min_silence`    | Integer | 2000    | Min silence duration (ms).      |
| `silence_thresh` | Integer | -63     | Audio level threshold (dBFS).   |
| `crossfade`      | Float   | 0.2     | Transition duration (seconds).  |
| `filler_words`   | Array   | [...]   | Words to transcribe and remove. |
| `remove_freeze`  | Boolean | False   | Enable frozen frame removal.    |
| `bg_method`      | Enum    | `rvm`   | `rvm` or `segmentation`.        |

## 3. Storage Structures

### 3.1 Directory Layout
- `web_app/static/uploads/`: Temporary storage for incoming videos. Deleted after processing.
- `web_app/static/outputs/`: Long-term storage for processed results.
