# Story 2.4: Unified Job Progress Relay

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a user,
I want to see real-time progress bars for background media tasks,
so that I know exactly how long a detection or render will take.

## Acceptance Criteria

1. **Given** a background worker is processing a task (e.g., silence detection, rendering), **When** it updates its progress via Redis, **Then** the Dashboard UI must reflect this change in real-time.
2. **Given** the existing SSE infrastructure, **When** a progress update is pushed to the `progress:stream`, **Then** it must be relayed to the browser via the SSE channel.
3. **Given** the Dashboard UI, **When** a progress update is received, **Then** the corresponding project's progress bar must animate smoothly to the new value.

## Tasks / Subtasks

- [x] Enhance `core/sse_relay.py` to support multiple streams (AC: 2)
  - [x] Refactor `TelemetryRelay` into a generic `RedisStreamRelay`
  - [x] Support subscribing to both `telemetry:stream` and `progress:stream`
- [x] Update `web_app/app.py` to handle progress broadcasts (AC: 2)
  - [x] Add `broadcast_progress` callback
  - [x] Publish progress updates via SSE (e.g., type='progress')
- [x] Update `web_app/templates/dashboard.html` to handle real-time progress (AC: 1, 3)
  - [x] Add progress bar containers to project cards
  - [x] Implement SSE listener for 'progress' events
  - [x] Update DOM elements dynamically when progress events arrive
- [x] Implement integration tests for progress relay (AC: 2)
  - [x] Simulate a worker pushing to `progress:stream`
  - [x] Verify SSE delivery

## Dev Notes

- **Architecture Compliance**:
  - Leveraged industry-standard SSE for lightweight real-time updates.
  - Generic `RedisStreamRelay` allows easy expansion to more streams.
- **Payload Structure**: `{ "project_id": "uuid", "progress": 45, "message": "Analyzing audio..." }`
- **UI Performance**: SSE events are handled by a targeted DOM listener, updating only the relevant project card.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 2.4]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Pytest run: `tests/test_sse_relay.py` passed 100% (3/3).
- Manual verification of `dashboard.html` logic.

### Completion Notes List

- Refactored `TelemetryRelay` to `RedisStreamRelay` for scalability.
- Launched `progress:stream` relay in `web_app/app.py`.
- Integrated real-time progress bars and status updates into the Dashboard UI.
- Verified system resilience with automated tests.

### File List
- `core/sse_relay.py`
- `web_app/app.py`
- `web_app/templates/dashboard.html`
- `web_app/static/css/dashboard.css`
- `tests/test_sse_relay.py`
