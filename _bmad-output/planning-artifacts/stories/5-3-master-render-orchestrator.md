# Story 5.3: Master Render Orchestrator

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As an editor, I want a single "Render" button that takes all my approved cuts and preset selections to produce a final high-quality video file, so that I can finish my project without manually running scripts.

## Acceptance Criteria

1. **Given** a project with reviewed cuts, **When** the "Start Render" button is clicked in the Portal, **Then** an API request must be sent to initiate the full render.
2. **Given** a render request, **When** processing, **Then** the system must utilize the selected preset (Speed/Quality) to assemble the final video.
3. **Given** a background render job, **When** progress changes, **Then** real-time SSE updates must keep the user Informed of the percentage completion.
4. **Given** a completed render, **When** successful, **Then** the final video path must be stored in the project database and displayed as a download link.

## Tasks / Subtasks

- [x] Implement `POST /api/projects/<project_id>/render` endpoint in `web_app/api.py` (AC: 1)
  - [x] Extract keep intervals from database (`CutCandidate` where `ignored=false`)
  - [x] Trigger the background render pipeline
- [x] Create `async_render_pipeline` background task (AC: 2)
  - [x] Call `process_video` with the correct intervals and parameters
  - [x] Broadcast progress via SSE (AC: 3)
  - [x] Store results in Project metadata (AC: 4)
- [x] Update `dashboard.html` UI
  - [x] Add Preset selection (Speed/Quality) to the review/render modal
  - [x] Hook up the "Start Render" button to the new endpoint
  - [x] Display render progress and completion status
- [x] Support project status transitions (ready -> rendering -> complete)

## Dev Notes

- **Master Orchestration**: The render pipeline now correctly bypasses detection by passing `forced_remove_intervals` to `processor.py`.
- **UI/UX**: Users can now choose between "Speed Mode" (GPU-prowess) and "Quality Mode" (MoviePy composition) directly from the dashboard.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 5.3]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Verified `processor.py` against `forced_remove_intervals`.
- Verified `api.py` endpoint registration.

### Completion Notes List

- Implemented `async_render_pipeline`.
- Added `/api/projects/<id>/render` endpoint.
- Updated Dashboard with Preset Select and Render Progress.
- Fixed `CutCandidate` field names bug in `api.py`.

### File List
- `web_app/api.py`
- `web_app/templates/dashboard.html`
- `processor.py`
