---
project_name: 'auto_video_editor'
user_name: 'Ben'
date: '2026-01-06'
sections_completed: ['technology_stack', 'language_rules', 'framework_rules', 'testing_rules', 'quality_rules', 'workflow_rules', 'anti_patterns']
status: 'complete'
rule_count: 26
optimized_for_llm: true
---

# Project Context for AI Agents

_This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss._

---

## Technology Stack & Versions

### Core Languages & Runtimes
- **Python**: >=3.10 (Managed via `uv`)
- **JavaScript**: Vanilla (ES6+) for Web Portal

### Primary Dependencies
- **Media Backend**: `ffmpeg` (with NVENC), `ffmpegcv` (GPU zero-copy), `av>=10.0.0` (PyAV)
- **AI/IL Logic**: `torch>=2.9.1`, `torchaudio>=2.9.1`, `transformers>=4.57.3`, `openai-whisper>=20250625`
- **Web & Dashboard**: `flask>=3.1.2`, `flask-sse`, `flask-cors>=6.0.2`
- **Logic**: `numpy>=2.2.6`, `librosa>=0.11.0`, `mediapipe>=0.10.9`, `pynvml` (HV Monitoring)

### Infrastructure
- **Broker**: Redis v7.0+ (Streams support mandatory)
- **Storage**: Hardware RAMDisk (`/dev/shm`) for volatile preview cache.

---

## Critical Implementation Rules

### 1. Hardware-Aware VRAM Guard (Strict Enforcement)
- **Monitoring**: Use `pynvml` for all VRAM health checks. Do NOT rely solely on `torch.cuda.memory_allocated()`.
- **Mandatory Buffer**: All media workers MUST check for a 1.5GB VRAM safety buffer before initialization.
- **Locking Pattern**: No worker may allocate GPU resources without acquiring the global Redis lock: `lock:vram_atomic`.
- **OOM Fail-Safe**: Catch `torch.cuda.OutOfMemoryError` and FFmpeg return codes. On OOM, log to telemetry and *immediately* release the Redis lock before exiting.

### 2. Zero-Copy Media Pipeline
- **GPU Performance**: Use `ffmpegcv` for frame extraction/rendering to keep pixel data on the GPU. Avoid host-RAM copies.
- **Audio Extract**: Use `PyAV` for low-overhead audio demuxing before transcription.
- **Whisper Optimization**: Prioritize `torch.compile` with "Regional Compilation" to reduce cold-start latency.

### 3. Naming & Case Conventions
- **Python**: Strict `snake_case` (e.g., `vram_guard.py`).
- **Redis Keys**: Hierarchical colons: `task:{type}:{uuid}`, `telemetry:{worker_type}`, `lock:vram_atomic`.
- **Frontend Events**: `kebab-case` (e.g., `render-progress`).

### 4. Communication & Workflow
- **Telemetry**: All worker heartbeats/progress MUST go to the Redis stream `telemetry:stream`.
- **Async Pattern**: Use `asyncio` for non-blocking telemetry relays in the Flask portal.
- **Environment**: `workers/base.py` MUST verify `/dev/shm` capacity (`>2GB` free) before initialization. Hard-fail (SystemExit) if invalid.
- **Resiliency**: Use circuit-breaker/retry decorators for Redis stream operations to handle transient loopback blips.

---

## Testing & Quality Rules
- **Lock-Lifecycle Tests**: All drivers must include a test simulating a hard crash (e.g. `sys.exit`) to verify Redis lock TTL/cleanup.
- **VRAM Mocking**: Mock "Low VRAM" states to verify worker queuing behavior.
- **Type Hinting**: Mandatory Python type hints for all public functions in `/core` and `/workers`.
- **Telemetry Schema**: Every `XADD` must include `msg_type` from `{vram|progress|error|log}`.
- **Commit Style**: Enforce Conventional Commits: `type(scope): description`.

---

## Project Structure & Boundaries
- **`/core`**: Shared utilities (Redis pools, VRAM guard, DB logic).
- **`/workers`**: Isolated Python processes/scripts for media tasks.
- **`/portal`**: Flask web application and SSE relay logic.
- **`/dev/shm`**: Primary volatile I/O boundary for sandbox previews.

---

## Usage Guidelines

**For AI Agents:**
- Read this file before implementing any code.
- Follow ALL rules exactly as documented; do not default to generic library patterns.
- When in doubt, prefer the more restrictive/hardware-safe option.

**For Humans:**
- Keep this file lean; remove rules that become "obvious" to the model over time.
- Update whenever the technology stack or VRAM budgeting logic changes.

**Last Updated**: 2026-01-06
