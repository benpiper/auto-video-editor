# Story 4.3: Instant Sensitivity Feedback Loop (Portal Integration)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As an editor, I want to adjust my detection settings and see an updated preview without a full page refresh, so that I can rapidly iterate.

## Acceptance Criteria

1. **Given** a project in the review state, **When** I click the "Generate Preview" button, **Then** the UI must trigger the `/api/projects/<id>/preview` endpoint.
2. **Given** an ongoing preview generation, **When** progress updates arrive via SSE, **Then** the dashboard must display a progress bar specifically for the preview task.
3. **Given** a finished preview, **When** complete, **Then** the UI must display a video player (or update an existing one) showing the 30s edited sandbox segment.
4. **Given** the preview player, **When** displayed, **Then** it must allow the user to verify the cuts before committing to a full render.

## Tasks / Subtasks

- [x] Update `dashboard.html` to include a Preview button and progress UI (AC: 1, 2)
- [x] Implement `startPreview` JavaScript function to call the API (AC: 1)
- [x] Add SSE handling for `type: preview` progress messages (AC: 2)
- [x] Create a "Preview Modal" or section in the review modal to play the generated video (AC: 3, 4)
- [x] Add an endpoint to serve the preview video from `/dev/shm` (AC: 3)

## Dev Notes

- **Dynamic SSE**: Updated `app.py` to support dynamic SSE event types based on the message data, enabling separate streams for `progress` (detection) and `preview`.
- **Sandbox Player**: The preview video is served directly from RAMDisk via a streaming endpoint, providing near-instant playback of the edited segment.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.3]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Verified UI elements in `dashboard.html`.
- Verified SSE routing in `app.py`.

### Completion Notes List

- Integrated preview generation and playback into the dashboard.
- Implemented real-time progress tracking for the preview task.
- Added `/api/projects/<id>/preview/stream` for efficient video serving.

### File List
- `web_app/templates/dashboard.html`
- `web_app/app.py`
- `web_app/api.py`
- `web_app/static/css/dashboard.css`
