# Story 1.2: Redis Client & Resilient Stream Wrappers

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want a unified Redis client with built-in circuit-breaker and retry logic,
so that inter-process communication remains stable during transient local network blips.

## Acceptance Criteria

1. **Given** a local Redis server (v7.0+), **When** `core/redis_client.py` is initialized, **Then** it must establish a connection pool with configurable timeouts (50ms - 200ms).
2. **Given** a transitient Redis failure, **When** a stream operation is attempted, **Then** a circuit-breaker must trigger to prevent thread/process blocking.
3. **Given** the requirement for decoupled communication, **When** calling the client, **Then** it must provide high-level `add_to_stream` and `read_stream` wrappers.
4. **Given** a development environment, **When** unit tests are run, **Then** Redis operations must be mocked using `fakeredis` to ensure tests are environment-agnostic.

## Tasks / Subtasks

- [x] Initialize `core/redis_client.py` (AC: 1)
  - [x] Define `RedisManager` class
  - [x] Implement connection pool with `redis.ConnectionPool`
  - [x] Configure `socket_timeout` and `socket_connect_timeout` (50ms-200ms)
- [x] Implement Circuit Breaker Logic (AC: 2)
  - [x] Implement/Integrate a decorator-based circuit breaker for Redis operations
  - [x] Define failure thresholds and reset timeouts
- [x] Implement Stream Wrappers (AC: 3)
  - [x] Implement `add_to_stream(stream_name, data)` using `XADD`
  - [x] Implement `read_stream(stream_name, last_id, count, block)` using `XREAD`
  - [x] Ensure wrappers use the circuit breaker protection
- [x] Update `pyproject.toml` dependencies (AC: 4)
  - [x] Add `redis` (if not already present)
  - [x] Add `fakeredis` to dev dependencies
- [x] Implement unit tests for `core/redis_client.py` (AC: 4)
  - [x] Test successful stream write/read
  - [x] Test circuit breaker trip on mocked connection error

## Dev Notes

- **Architecture Compliance**:
  - Redis Streams are the primary IPC for telemetry and tasking.
  - Architecture specifies Redis v7.0+.
- **Resilience Strategy**:
  - Use `circuitbreaker` library or a robust custom class.
  - NFR2 (<45s latency) relies on fast Redis responses; avoid long block timeouts in `XREAD`.
- **Naming Conventions**:
  - Redis keys: `telemetry:stream`, `task:whisper:{uuid}`, etc. (Hierarchical colons).

### Project Structure Notes

- Core utilities reside in `core/`.
- Connection parameters should eventually be pulled from `.env` via `core/config.py` if available, or defaulted.

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Structural Patterns]
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.2]
- [Source: _bmad-output/planning-artifacts/project-context.md#3. Naming & Case Conventions]

## Senior Developer Review (AI)

**Review Date:** 2026-01-07
**Outcome:** Approved (after fixes)

### Action Items
- [x] [HIGH] Security: Password Leakage in Logs (Masked URL in logs).
- [x] [HIGH] Architecture: Raw Data Leakage (Normalized `read_stream` output).
- [x] [MEDIUM] Idle Connection Decay (Added `health_check_interval`).
- [x] [MEDIUM] Inconsistent Empty Responses (Fixed with empty list return).
- [x] [LOW] Hardcoded Resilience Parameters (Moved to global constants).
- [x] [LOW] Missing Type Hinting (Refined return types).

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Pytest run: `tests/test_redis_client.py` passed 100% (3/3).
- Circuit breaker threshold validation: Confirmed trip after 5 failures in tests.

### Completion Notes List

- Implemented `RedisManager` in `core/redis_client.py`.
- Integrated `circuitbreaker` package for resilient operations.
- Added `add_to_stream` and `read_stream` high-level wrappers.
- Configured connection pool with AC-compliant timeouts (0.2s socket, 0.1s connect).
- Verified everything with `fakeredis` unit tests.

### File List
- `core/redis_client.py`
- `pyproject.toml`
- `tests/test_redis_client.py`
