# Story 1.1: Environment Initialization & RAMDisk Setup

Status: ready-for-dev

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

- [ ] Create `scripts/setup_ramdisk.sh` (AC: 1)
  - [ ] Implement check for existing `/dev/shm` mount
  - [ ] Implement logic to resize or mount at least 4GB
  - [ ] Add sudo validation check
- [ ] Implement `workers/base.py` (AC: 2)
  - [ ] Define `BaseWorker` class
  - [ ] Implement `verify_environment()` with `/dev/shm` capacity check
  - [ ] Ensure `SystemExit` on failure with descriptive error
- [ ] Update `pyproject.toml` dependencies (AC: 3)
  - [ ] Add `nvidia-ml-py`
  - [ ] Remove `pynvml` if present

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

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

### Completion Notes List

### File List
- `scripts/setup_ramdisk.sh`
- `workers/base.py`
- `pyproject.toml`
