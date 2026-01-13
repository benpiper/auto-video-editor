# Story 2.2: Fast Media Ingest & Metadata Extraction

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a user,
I want to register local video files and see their metadata instantly,
so that I can begin the cleaning process without manually entering specs.

## Acceptance Criteria

1. **Given** a valid path to a 4K/1080p .mp4 or .mov file, **When** I "Ingest" the file via the Flask portal route, **Then** the system must use `PyAV` (av) to extract duration, resolution, frame rate, and audio/video stream parameters.
2. **Given** successfully extracted metadata, **When** the ingest process completes, **Then** the metadata must be stored in the project's database record and returned to the user.
3. **Given** an invalid file path or unsupported format, **When** ingestion is attempted, **Then** the system must return a clear error message.

## Tasks / Subtasks

- [x] Implement `core/media_info.py` (AC: 1)
  - [x] Use `av` to open video container
  - [x] Extract: duration, width, height, fps, video_codec, audio_codec
  - [x] Format as a serializable dictionary
- [x] Implement Ingest API Route in `web_app/api.py` (AC: 1, 2, 3)
  - [x] Create `/projects/ingest` POST endpoint
  - [x] Handle input file path
  - [x] Call `media_info` extraction
  - [x] Create `Project` record in database with extracted specs
- [x] Implement unit tests for metadata extraction
  - [x] Test with a real sample video file
  - [x] Test error handling for missing/corrupt files

## Dev Notes

- **Architecture Compliance**:
  - Use `PyAV` as requested (performance sensitive).
  - Store metadata in the `Project.metadata` JSON field.
- **Dependencies**:
  - `av` is already in `pyproject.toml`.

### Project Structure Notes

- Media utility in `core/media_info.py`.
- API endpoint in `web_app/api.py`.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 2.2]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

### Completion Notes List

### File List
- `core/media_info.py`
- `web_app/api.py`
- `tests/test_media_info.py`
