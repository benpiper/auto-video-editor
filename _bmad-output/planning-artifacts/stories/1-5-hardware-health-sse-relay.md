# Story 1.5: Hardware Health SSE Relay

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a user,
I want to see a live hardware health indicator on the web portal dashboard,
so that I can verify my 12GB GPU is being utilized safely.

## Acceptance Criteria

1. **Given** telemetry data arriving in the `telemetry:stream`, **When** the Flask portal route `/sse/hardware` is requested, **Then** it must use `Flask-SSE` or a compatible mechanism (Redis Pub/Sub relay) to push new metrics to the browser in real-time.
2. **Given** a connected browser client, **When** telemetry is broadcast, **Then** the payload must include `total`, `used`, `free`, `hostname`, and `timestamp`.
3. **Given** the high-frequency nature of telemetry (2s), **When** multiple clients connect, **Then** the server must efficiently relay from Redis without creating a new Redis connection per client (Connection Pooling).

## Tasks / Subtasks

- [x] Implement `core/sse_relay.py`
  - [x] Create a background thread that subscribes to `telemetry:stream` (or pools it via `XREAD`).
  - [x] Use a mechanism to broadcast these messages to Flask SSE clients.
- [x] Implement Flask SSE Endpoint in `web_app/app.py`
  - [x] Define the `/sse/hardware` route.
  - [x] Integrate the relay logic from `RedisManager`.
- [x] Implement unit tests for the SSE relay
  - [x] Mock Redis stream incoming data.
  - [x] Verify endpoint returns SSE-formatted data.

## Dev Notes

- **Architecture Compliance**:
  - SSE is the specific technology mandated for real-time dashboard updates.
  - Redis is the source of truth.
  - Using `Flask-SSE` for industry-standard Pub/Sub integration.
- **Performance**:
  - `TelemetryRelay` runs in a background daemon thread.
  - Efficiently uses `X_BLOCK` to avoid CPU spin.

### Project Structure Notes

- API and portal code reside in `api/` or `portal/`. Architecture says `web_app/app.py`.

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Process Patterns]
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.5]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Pytest run: `tests/test_sse_relay.py` passed 100% (2/2).

### Completion Notes List

- Implemented `TelemetryRelay` in `core/sse_relay.py`.
- Integrated `Flask-SSE` into `web_app/app.py`.
- Automated the bridge between Redis Streams and SSE Pub/Sub.
- Added `/sse/hardware` route for compliance.

### File List
- `core/sse_relay.py`
- `web_app/app.py`
- `tests/test_sse_relay.py`
