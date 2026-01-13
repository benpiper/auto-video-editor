# Story 1.1: Environment Initialization & RAMDisk Setup

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a system orchestrator,
I want to automatically configure the Linux RAMDisk (/dev/shm) and verify its capacity,
so that all volatile media processing stays on high-speed hardware with zero SSD wear.

## Acceptance Criteria

1. **Given** a Linux host with root/sudo access for initial setup, **When** the `scripts/setup_ramdisk.sh` is executed, **Then** it must mount or verify a RAMDisk at `/dev/shm` with at least 4GB allocation.
2. **Given** the system is running, **When** the `workers/base.py` `verify_environment()` method is called, **Then** it must hard-fail (SystemExit) if `/dev/shm` free space is `< 2GB`.
3. **Given** a new environment, **When** dependencies are installed, **Then** `nvidia-ml-py` must be used instead of the deprecated `pynvml`.

## Tasks / Subtasks

- [x] Create `scripts/setup_ramdisk.sh` (AC: 1)
  - [x] Implement check for existing `/dev/shm` mount
  - [x] Implement logic to resize or mount at least 4GB
  - [x] Add sudo validation check
- [x] Implement `workers/base.py` (AC: 2)
  - [x] Define `BaseWorker` class
  - [x] Implement `verify_environment()` with `/dev/shm` capacity check
  - [x] Ensure `SystemExit` on failure with descriptive error
- [x] Update `pyproject.toml` dependencies (AC: 3)
  - [x] Add `nvidia-ml-py`
  - [x] Remove `pynvml` if present

## Dev Notes

- **Architecture Compliance**:
  - High-Speed RAMDisk (`/dev/shm`) is critical for NFR2 (<45s latency).
  - Use `uv` for all dependency management.
  - Architecture specifies Python 3.10+.
- **Hardware Limit**: 12GB NVIDIA VRAM (governed by later stories).
- **Communication Patterns**: Not directly involved in this story, but `workers/base.py` will eventually host the Redis client initialization.

### Project Structure Notes

- Scripts belong in `scripts/`.
- Worker base classes belong in `workers/`.
- Configuration should eventually reside in `core/config.py` (referenced in later stories).

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Hardware Performance & Memory Strategy]
- [Source: _bmad-output/planning-artifacts/prd.md#Non-Functional Requirements]
- [Source: Technical Research - nvidia-ml-py is the official replacement for pynvml]

## Senior Developer Review (AI)

**Review Date:** 2026-01-07
**Outcome:** Approved (after fixes)

### Action Items
- [x] [HIGH] Brittle Test Coverage: Mock `os.path.exists` and add missing path case.
- [x] [MEDIUM] Logging Side-Effects: Move `basicConfig` out of module level in `base.py`.
- [x] [MEDIUM] Assumptive Resizing: Improve `setup_ramdisk.sh` mount logic.
- [x] [LOW] Missing Core Dependency: Add `redis` to `pyproject.toml`.
- [x] [LOW] Shell Portability: Use `awk` for robust parsing in shell scripts.
- [x] [LOW] Observability Gap: Log total and used capacity in `verify_environment`.

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Pytest run: `tests/test_base_worker.py` passed 100% (3/3 tests).
- Script execution: `scripts/setup_ramdisk.sh` verified on local environment.

### Completion Notes List

- Created `scripts/setup_ramdisk.sh` with robust size checking and mounting logic.
- Implemented `workers/base.py` with clean logging and environment verification.
- Addressed all code review findings (1 High, 2 Medium, 3 Low).
- Verified with unit tests.

### File List
- `scripts/setup_ramdisk.sh`
- `workers/base.py`
- `workers/__init__.py`
- `pyproject.toml`
- `tests/test_base_worker.py`

## Change Log

- **2026-01-07**: Addressed code review findings - 6 items resolved.
- **2026-01-06**: Initial implementation.
