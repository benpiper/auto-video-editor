# Story 4.1: Sandbox Segment Extractor (ffmpegcv)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a system, I want to isolate the first 30 seconds of high-resolution video using zero-copy extraction, so that I can provide a representative sample for preview.

## Acceptance Criteria

1. **Given** a project with an ingested video file, **When** a Sandbox Preview is requested, **Then** the system must extract the first 30 seconds.
2. **Given** the extraction process, **When** executed, **Then** it must utilize `ffmpegcv` or efficient FFmpeg zero-copy (stream copy) methods to ensure speed.
3. **Given** performance requirements, **When** extracting 30s from a 4K video, **Then** it must complete in `< 5 seconds`.
4. **Given** storage preferences, **When** extracted, **Then** the segment must be stored in `/dev/shm` (RAMDisk) for subsequent processing.

## Tasks / Subtasks

- [x] Implement `extract_sandbox_segment` in `processor.py` (AC: 1, 2)
  - [x] Use FFmpeg stream copying for zero-copy performance (AC: 3)
  - [x] Support `/dev/shm` for high-speed scratch space (AC: 4)
- [x] Integrate into the API pipeline
  - [x] Add `POST /api/projects/<project_id>/preview` endpoint
- [x] Implement unit tests for segment extraction
  - [x] Verify duration and storage integrity (AC: 3)

## Dev Notes

- **Zero-Copy**: Achieved via `ffmpeg -c copy`, ensuring sub-second extraction even for 4K content.
- **RAMDisk**: Defaults to `/dev/shm` for sandbox storage to enable near-instant read/write for the assembly phase.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.1]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Passed `tests/test_sandbox_extract.py`.

### Completion Notes List

- Implemented `extract_sandbox_segment` in `processor.py`.
- Added `/preview` endpoint and `async_preview_pipeline` in `api.py`.
- Verified 30s cut accuracy and speed.

### File List
- `processor.py`
- `web_app/api.py`
- `tests/test_sandbox_extract.py`
