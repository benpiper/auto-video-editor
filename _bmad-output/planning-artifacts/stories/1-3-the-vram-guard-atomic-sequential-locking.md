# Story 1.3: The VRAM Guard (Atomic Sequential Locking)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a system,
I want to enforce a global hardware lock for high-VRAM tasks,
so that only one resource-intensive worker (Whisper or NVENC) is active at a time.

## Acceptance Criteria

1. **Given** multiple workers contending for the GPU, **When** a worker attempts to enter a `with vram_guard():` context block, **Then** it must successfully acquire the Redis key `lock:vram_atomic`.
2. **Given** a worker holding the lock, **When** the work is complete or an exception occurs, **Then** the lock must be released immediately (Context Manager `__exit__`).
3. **Given** a hardware lock, **When** the lock is acquired, **Then** it must have a 60s inactivity timeout (TTL) to prevent system deadlocks in case of a crash.
4. **Given** the Redis cluster, **When** the lock is held, **And** the worker is still active, **Then** the lock TTL should be extensible if the work exceeds 60s.

## Tasks / Subtasks

- [x] Implement `core/vram_guard.py` (AC: 1, 2, 3)
  - [x] Define `VRAMGuard` context manager class
  - [x] Implement `__enter__` to acquire Redis lock `lock:vram_atomic`
  - [x] Use `nx=True` (Set if Not Exists) and `px=60000` (60s TTL)
  - [x] Implement retry loop/blocking logic for lock acquisition
  - [x] Implement `__exit__` to release the lock safely
- [x] Implement Lock Heartbeat/Extension (AC: 4)
  - [x] Implement a mechanism to extend the lock TTL if the context block is still running
- [x] Implement integration tests for locking functionality
  - [x] Test lock acquisition and release
  - [x] Test lock contention (second worker waits for first)
  - [x] Test automatic cleanup/TTL expiration

## Dev Notes

- **Architecture Compliance**:
  - Global Redis lock is mandatory for all GPU-intensive tasks.
  - Key name MUST be `lock:vram_atomic`.
- **Resilience**:
  - Redlock algorithm principles applied via atomic Lua scripts for release/extension.
  - Guaranteed atomic release (only owner can release).
- **Deadlock Prevention**:
  - The 60s TTL is critical.
  - Heartbeat ensures long jobs don't lose the lock prematurely.

### Project Structure Notes

- VRAM Guard resides in `core/`.
- Uses `RedisManager` from `core/redis_client.py`.

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Process Patterns]
- [Source: _bmad-output/planning-artifacts/project-context.md#1. Hardware-Aware VRAM Guard]
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.3]

## Senior Developer Review (AI)

**Review Date:** 2026-01-08
**Outcome:** Approved (after fixes)

### Action Items
- [x] [HIGH] Infinite Hang Risk: Added `max_wait_ms` acquisition timeout.
- [x] [HIGH] Resource Leak: Improved heartbeat thread cleanup behavior.
- [x] [MEDIUM] Architecture: VRAM Safety Buffer (Added structural hooks).
- [x] [MEDIUM] Brittle Tests: Improved test robustness with timeout cases.
- [x] [LOW] Logging Noise: Reduced heartbeat log level to DEBUG.
- [x] [LOW] Configuration: Moved intervals/TTLs to configurable environment constants.

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Pytest run: `tests/test_vram_guard.py` passed 100% (4/4).
- Heartbeat confirmed: TTL extension verified in unit tests.

### Completion Notes List

- Implemented `VRAMGuard` context manager in `core/vram_guard.py`.
- Developed atomic Lua scripts for safe lock release and TTL extension.
- Integrated background heartbeat thread for long-running task support.
- Verified contention, cleanup, and heartbeat logic with unit tests.

### File List
- `core/vram_guard.py`
- `tests/test_vram_guard.py`
- `pyproject.toml`
