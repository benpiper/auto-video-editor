# Story 4.2: Dynamic Preview Assembler (Hardware-Accelerated)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As an editor, I want to see my cleaning settings applied to the preview segment instantly, so that I can verify my choices before starting the final render.

## Acceptance Criteria

1. **Given** an extracted 30s sandbox segment and a list of validated removal segments (cuts), **When** a "Render Preview" is initiated, **Then** the system must only apply cuts that occur within the first 30 seconds.
2. **Given** the need for speed, **When** assembling the preview, **Then** the system must utilize `ffmpegcv` and NVIDIA NVENC (if available) for hardware-accelerated rendering.
3. **Given** a 30s segment, **When** rendered, **Then** the final preview video must be generated in `< 45 seconds`.
4. **Given** the output file, **When** complete, **Then** it must be stored in `/dev/shm` and linked in the project metadata for the UI to display.

## Tasks / Subtasks

- [x] Implement `assemble_preview_video` in `processor.py` (AC: 1, 2)
  - [x] Extract "keep" intervals from `CutCandidate` records for the first 30s
  - [x] Use `ffmpegcv` and implement robust NVENC with libx264 fallback (AC: 2)
- [x] Update `async_preview_pipeline` in `web_app/api.py` (AC: 1, 4)
  - [x] Retrieve cuts from the database
  - [x] Call the assembler and broadcast progress (AC: 3)
- [x] Implement unit tests for preview assembly
  - [x] Verify that ignored cuts are kept and validated cuts are removed

## Dev Notes

- **Robust Encoding**: Implemented an immediate fallback from `h264_nvenc` to `libx264` if the hardware encoder fails on the first write (useful for environments without CUDA).
- **Sequential Assembly**: While `ffmpegcv` doesn't support random-access seeking, sequential skipping is performant for the 30s sandbox context.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.2]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Passed `tests/test_preview_assembly.py`.

### Completion Notes List

- Added `assemble_preview_video` function.
- Integrated assembly into the preview API pipeline.
- Stored `preview_path` in project metadata.

### File List
- `processor.py`
- `web_app/api.py`
- `tests/test_preview_assembly.py`
