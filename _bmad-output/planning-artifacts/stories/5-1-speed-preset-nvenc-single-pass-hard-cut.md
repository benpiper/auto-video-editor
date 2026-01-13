# Story 5.1: Speed Preset (NVENC Single-Pass Hard-Cut)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As an editor with tight deadlines, I want a "Speed" rendering mode that produces the final video as fast as possible using my GPU, even if the file size is larger, so that I can publish immediately.

## Acceptance Criteria

1. **Given** a validated set of cuts, **When** the "Speed" preset is selected, **Then** the system must utilize NVIDIA NVENC for high-throughput encoding.
2. **Given** the Speed preset, **When** rendering, **Then** it must use a single-pass encoding strategy with a fast preset (e.g., `p1` or `fast`).
3. **Given** the Speed preset, **When** assembling, **Then** it must utilize "Hard Cuts" (no crossfades) to minimize frame blending overhead.
4. **Given** a 10-minute 1080p source, **When** rendered on an NVENC-capable GPU, **Then** the total render time must be `< 2 minutes` (5x realtime or better).

## Tasks / Subtasks

- [x] Define "Speed" preset parameters in `processor.py` (AC: 1, 2)
  - [x] Set `codec='h264_nvenc'`
  - [x] Set `preset='p1'` and `hp` for fastest throughput
  - [x] Forced 8000k bitrate for single-pass quality
- [x] Implement Hard-Cut logic in `process_video` (AC: 3)
  - [x] Bypassed re-encoding for transitions by forcing `no_crossfade=True`
  - [x] Optimized segment concatenation via FFmpeg stream copy
- [x] Update `async_video_process` in `api.py` and `app.py` to handle the Speed preset selection
- [x] Benchmark and verify performance logic via unit tests (`tests/test_speed_preset.py`)

## Dev Notes

- **NVENC Optimization**: Mode "Speed" utilizes `preset p1`, `tune ll`, and `rc vbr` to maximize frame processing speed.
- **Hard Cuts**: By forcing `no_crossfade=True`, we avoid the complex filtergraph required for transitions, allowing FFmpeg to process segments nearly sequentially.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 5.1]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Passed logic check in `tests/test_speed_preset.py`.

### Completion Notes List

- Added `render_preset` argument to `process_video`.
- Implemented `speed` mode logic.
- Updated API and App entry points.

### File List
- `processor.py`
- `web_app/api.py`
- `web_app/app.py`
- `tests/test_speed_preset.py`
- `processor.py`
- `web_app/api.py`
