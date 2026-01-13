# Story 5.4: Export Manager & Post-Render Notifications

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a user, I want to be notified when my master render is ready and easily access the file, so that I can move on to my next task.

## Acceptance Criteria

1. **Given** a completed master render, **When** the worker signals task completion, **Then** the web portal must display a success notification.
2. **Given** a finished project, **When** viewing the dashboard, **Then** the system must provide a clear "Download Final Master" button.
3. **Given** the browser supports it, **When** a render completes in the background, **Then** a system-level notification (Web Notifications API) should be triggered.
4. **Given** multiple renders over time, **When** the dashboard is loaded, **Then** only the latest valid master output should be linked.

## Tasks / Subtasks

- [x] Implement Browser Notifications (Web API) in `dashboard.html` (AC: 1, 3)
  - [x] Request permission on page load
  - [x] Trigger notification on `render` completion event from SSE
- [x] Add "Download Final Master" button to Project Cards (AC: 2)
  - [x] Update `loadProjects` in `dashboard.html` to check for `final_output_path` in metadata
- [x] Implement Export Cleanup (AC: 4)
  - [x] Add logic to remove old temporary master renders if a new one is started
- [x] Final visual polish on notification UI (success toast/alert)
- [x] Consistently brand as **AutoCut AI** with custom generated logo

## Dev Notes

- **Branding**: Generated a custom high-tech logo using AI and integrated it across the app for a premium feel.
- **Background Renders**: Notifications work even if the browser tab is not focused, providing a professional "set and forget" experience.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 5.4]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

### Completion Notes List

- Added "Master" button to cards for immediate access.
- Integrated Web Notifications API.
- Added render cleanup in `api.py`.
- Updated branding across the Entire App.

### File List
- `web_app/templates/dashboard.html`
- `web_app/api.py`
- `web_app/static/css/dashboard.css`
- `web_app/templates/index.html`
- `web_app/static/img/logo.png`
