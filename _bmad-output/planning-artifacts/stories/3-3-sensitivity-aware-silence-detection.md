# Story 3.3: Sensitivity-Aware Silence Detection

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As an editor,
I want to identify boring silences based on my own threshold settings,
so that I can control the pacing of the edited video.

## Acceptance Criteria

1. **Given** a mono audio file and a user-defined dB threshold, **When** the silence detection logic runs, **Then** it must identify all segments below the threshold for more than a set duration (default 2 seconds).
2. **Given** a large video file, **When** detecting silence, **Then** the process must report progress periodically to the dashboard.
3. **Given** different types of content (e.g., quiet room vs. noisy background), **When** configured, **Then** the system must allow the user to override the `silence_thresh` and `min_silence_len` via the API.

## Tasks / Subtasks

- [x] Optimize `detect_silence` in `processor.py` for large files
  - [x] Implement chunked processing to support progress reporting (AC: 2)
  - [x] Integrate progress callback into the loop
- [x] Pass through configurable parameters from the API to simple silence detection (AC: 3)
- [x] Implement unit tests for silence detection with different thresholds
  - [x] Test with a synthetic audio containing known silence gaps (AC: 1)

## Dev Notes

- **Chunked Processing**: Subdivides audio into 30s chunks to provide near-real-time progress feedback on the dashboard.
- **Configurable**: Thresholds are passed from the API throughout the pipeline.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 3.3]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Passed `tests/test_silence_detection.py`.

### Completion Notes List

- Refactored `detect_silence` for chunking.
- Added progress reporting logic to `process_video`.

### File List
- `processor.py`
- `tests/test_silence_detection.py`
