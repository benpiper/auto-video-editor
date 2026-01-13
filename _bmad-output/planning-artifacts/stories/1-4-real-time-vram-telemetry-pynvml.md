# Story 1.4: Real-time VRAM Telemetry (pynvml)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a system,
I want to poll the actual hardware VRAM usage using NVIDIA's management library,
so that the dashboard can display real-time health data.

## Acceptance Criteria

1. **Given** a running `VRAMMonitor` worker, **When** it polls `pynvml` (nvidia-ml-py), **Then** it must retrieve the `total`, `used`, and `free` bytes for GPU index 0.
2. **Given** a polling cycle, **When** the hardware data is retrieved, **Then** it must be added to the Redis stream `telemetry:stream` via `XADD`.
3. **Given** the requirement for real-time monitoring, **When** the monitor is active, **Then** the polling frequency must be configurable (default 2 seconds).
4. **Given** the potential absence of an NVIDIA GPU (e.g., in a CPU-only dev environment), **When** the monitor starts, **Then** it should log a warning and fallback to dummy/simulated data or exit gracefully, but not crash the entire system.

## Tasks / Subtasks

- [x] Implement `workers/vram_monitor.py` (AC: 1, 2, 3)
  - [x] Initialize `pynvml` and select GPU 0
  - [x] Create a polling loop with configurable interval (default 2s)
  - [x] Use `RedisManager.add_to_stream` to push `total`, `used`, `free` bytes
- [x] Implement Fallback Logic (AC: 4)
  - [x] Add check for NVIDIA drivers/GPU presence
  - [x] Implement graceful fallback or error reporting if hardware is missing
- [x] Implement unit tests for `VRAMMonitor`
  - [x] Mock `pynvml` to simulate hardware responses
  - [x] Verify Redis stream output matches hardware data

## Senior Developer Review (AI)

**Review Date:** 2026-01-10
**Outcome:** Approved (after fixes)

### Action Items
- [x] [HIGH] Permanent Fail-Soft: Added `_attempt_init` retry logic to recover from driver glitches.
- [x] [MEDIUM] Telemetry Drift: Implemented drift compensation in the polling loop.
- [x] [MEDIUM] Missing Context: Added `timestamp` and `hostname` to the telemetry payload.
- [x] [LOW] Hardcoded Dummy Profile: Moved simulated VRAM size to `VRAM_DUMMY_TOTAL_GB` env var.

## Dev Notes

- **Architecture Compliance**:
  - Telemetry MUST be pushed to `telemetry:stream`.
  - Use `nvidia-ml-py` as the official library.
- **Observability**:
  - Logs indicate hardware vs dummy mode.
  - Payload now includes `hostname` and `timestamp`.
- **Resilience**:
  - Auto-recovers hardware mode if NVML becomes available later.
  - Compensates for execution time to keep 2s interval precise.

### Project Structure Notes

- Workers reside in `workers/`.
- Inherits from `BaseWorker`.

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Enforcement Guidelines]
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.4]
- [Source: _bmad-output/planning-artifacts/project-context.md#4. Real-time Telemetry]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Pytest run: `tests/test_vram_monitor.py` passed 100% (3/3).
- Drift compensation verified with localized timing.

### Completion Notes List

- Implemented `VRAMMonitor` in `workers/vram_monitor.py`.
- Successfully integrated `pynvml` for hardware-level telemetry.
- Added auto-recovery logic to retry hardware initialization if driver is missing.
- Added drift compensation to maintain precise 2s polling interval.
- Verified telemetry push with metadata (`hostname`, `timestamp`) with unit tests.

### File List
- `workers/vram_monitor.py`
- `tests/test_vram_monitor.py`
