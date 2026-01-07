---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
workflowType: 'architecture'
project_name: 'auto_video_editor'
user_name: 'Ben'
date: '2026-01-05'
status: 'complete'
completedAt: '2026-01-05'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**
Architecture must support an 18-point capability contract focusing on a multi-stage media pipeline (Detection -> Sandbox Preview -> Master Render). Critical capabilities include verbatim filler word detection, a low-latency 30s preview engine, and a hardware-aware "VRAM Guard."

**Non-Functional Requirements:**
- **Performance**: <0.5 render ratio for draft exports; <45s for sandbox previews.
- **Stability**: Mandatory 1.5GB VRAM buffer; isolated failure of AI workers; sequential staggering of high-memory tasks.
- **Security**: Strict Local-Only operations (no external API calls for processing).

**Scale & Complexity:**
- Primary domain: AI Media Processing / Local Full-stack
- Complexity level: Medium (Performance-heavy hardware orchestration)
- Estimated architectural components: ~6 (Web Portal, Redis/Broker, Ingest Worker, AI Detection Worker, Preview Worker, Master Render Worker)

### Technical Constraints & Dependencies
- **Hardware**: Dedicated 12GB NVIDIA GPU (Linux host).
- **Inference**: Existing Whisper model (integrated in Python/Torch).
- **Processing**: FFmpeg with NVENC/NVDEC hardware blocks; `ffmpegcv` for zero-copy transfers.

### Cross-Cutting Concerns Identified
- **VRAM Budgeting**: Every component must be aware of and respect the global VRAM budget to prevent CPU-fallback.
- **Task Pre-emption**: High-priority "Sandbox" requests may need to signal or pause long-running "Master Render" tasks.
- **Real-time Observability**: Consistent telemetry from workers to the web portal for the "Hardware Health Monitor."

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
- **VRAM Guard Strategy**: Atomic Sequential Locking via Redis. Only one high-memory worker (Whisper/NVENC) active at once.
- **Task Dispatch Pattern**: Unified Redis Streams (XADD/XREAD) for both tasking and telemetry.
- **Processing Storage**: High-Speed RAMDisk (`/dev/shm`) for all intermediate preview segments and isolated audio.

**Important Decisions (Shape Architecture):**
- **Telemetry State**: Stateful Redis Streams to support dashboard refresh and history playback.
- **Media Engine**: `ffmpegcv` for zero-copy GPU frame extraction and rendering.
- **Audio Prep**: `PyAV` for low-overhead audio stream demuxing before Whisper processing.

**Deferred Decisions (Post-MVP):**
- **Still Segment Logic**: Deferred to Phase 2 to focus on audio-cleaning stability.
- **Multi-GPU Orchestration**: Deferred until user scales beyond single 12GB host.

### Data Architecture
- **Worker Communication**: Redis Streams (v7.0+ recommended).
- **Intermediate Assets**: Volatile storage in `/dev/shm/{project_id}/` for previews; system-managed temp paths for Master Renders.
- **Rationale**: Minimal SSD wear and maximum I/O speed for the "Alex" efficiency journey.

### API & Communication Patterns
- **Dispatch Pattern**: Consumer Groups (Redis) to allow future worker scaling while maintaining atomic serial execution in MVP.
- **Telemetry**: Server-Sent Events (SSE) relaying Redis Stream events to the browser dashboard every 2s.

### Infrastructure & Deployment
- **Runtime**: Python 3.10 with `asyncio`.
- **Worker Execution**: Sequential execution enforced by a Global Redis Lock (`vram_active_lock`).
- **Monitoring**: Real-time VRAM telemetry via `pynvml` or `nvidia-smi` exported to the SSE stream.

### Decision Impact Analysis

**Implementation Sequence:**
1. Redis Stream schema & Task Dispatcher.
2. RAMDisk initialization & PyAV Audio extractor.
3. Atomic Lock implementation (The VRAM Guard).
4. `ffmpegcv` Renderer with NVENC support.

**Cross-Component Dependencies:**
The **VRAM Guard** is the central dependency. The Web Portal will display "Hardware Locked" status when a Master Render is active, preventing Sandbox requests from triggering OOM crashes.

## Starter Pattern Evaluation

### Primary Technology Domain
**AI-Driven Media Processing & Hardware Governance** based on project requirements for 4K video cleaning and 12GB VRAM stability.

### Starter Options Considered
- **FFmpeg Subprocess Wrappers** (Existing): High overhead, difficult to manage zero-copy frame transfers.
- **PyAV**: Direct C-bindings, excellent for audio/metadata but requires careful management for GPU-accelerated video streams.
- **ffmpegcv**: Optimized for deep learning pipelines, provides direct GPU acceleration for decoding and pixel format conversion (YUV -> RGB).
- **WebSockets (SocketIO)**: Bidirectional, but overkill and statefully complex for unidirectional hardware monitoring.
- **Flask-SSE**: Lightweight, native browser reconnection, perfect for unidirectional "Hardware Health" telemetry.

### Selected Patterns: The "Accelerated Media & SSE" Foundation

**Rationale for Selection:**
To achieve the **Experience-First MVP** on a 12GB budget, we must eliminate memory-copy bottlenecks. `ffmpegcv` ensures that frames stay on the GPU as long as possible, while `Flask-SSE` provides a stable, low-latency monitoring channel without the overhead of WebSocket state management.

**Initialization Command:**
```bash
pip install ffmpegcv flask-sse pyav
```

**Architectural Decisions Provided by Patterns:**

**Language & Runtime:**
Continued use of Python 3.10+, utilizing `asyncio` for non-blocking worker telemetry to Redis.

**Hardware Acceleration:**
Mandatory use of `h264_nvenc` and `hevc_nvenc` via `ffmpegcv` to offload pixel conversion and encoding from the CPU.

**Observation Pattern:**
Unidirectional SSE stream for the "Hardware Health Monitor," refreshing every 2s as per NFR requirements.

**Metadata Handling:**
`PyAV` will be used for high-precision audio stream extraction to feed the Whisper filler-word detection module.
## Implementation Patterns & Consistency Rules

### Pattern Categories Defined
**Conflict Points**: 4 primary areas identified: Naming, Payload Contracts, Error Recovery, and Project Organization.

### Naming Patterns
- **Python Code**: Strict `snake_case` for variables, functions, and filenames (e.g., `vram_monitor.py`).
- **Redis Keys**: Hierarchical colon-separated keys: `task:{type}:{id}`, `telemetry:{worker_type}`, `lock:vram_atomic`.
- **Frontend/SSE Events**: `kebab-case` for event identifiers (e.g., `render-progress`).

### Communication Patterns (Redis Streams)
- **Task Payload Structure**: 
  ```json
  { "task_id": "uuid", "input_path": "/dev/shm/...", "settings": { "silence_db": -40 } }
  ```
- **Telemetry Payload Structure**: 
  ```json
  { "project_id": "uuid", "msg_type": "vram|progress|error", "data": "value", "ts": "ISO-8601" }
  ```

### Structural Patterns
- **Worker Logic**: Contained within `/workers`. Each worker must implement a `cleanup` method to release locks on failure.
- **Shared Utilities**: Centralized in `/core` for Redis connection pools and VRAM monitoring.
- **Telemetry Relay**: Flask routes exclusively in `/portal` to aggregate Redis Stream data to SSE.

### Process Patterns (Error & Loading)
- **OOM Handling**: Workers MUST catch `torch.cuda.OutOfMemoryError` or FFmpeg exit codes, log to the stateful stream, and release the global `vram_active_lock`.
- **Atomic Locking**: No task may begin `ffmpegcv` or `Whisper` initialization without successfully acquiring the Redis Lock.
- **Refresh Protection**: Telemetry is streamed from Redis history, ensuring the Dashboard remains consistent across page refreshes.

### Enforcement Guidelines
**All AI Agents MUST:**
- Use `pynvml` or `nvidia-smi` for all VRAM reporting to ensure consistency in the health monitor.
- Enforce the 1.5GB VRAM safety buffer before initiating new workers.
- Default to `h264_nvenc` for all preview-class renders.

## Project Structure & Boundaries

### Complete Project Directory Structure

```text
auto_video_editor/
├── main.py                 # Entry point / orchestrator
├── requirements.txt
├── config.yaml             # VRAM limits: 12GB total, 1.5GB buffer
├── .env                    # Redis URL, RAMDisk path (/dev/shm)
├── core/
│   ├── redis_client.py     # Stream XADD/XREAD wrappers
│   ├── vram_guard.py       # Global Redis Lock + pynvml monitoring
│   └── db.py               # Project/Task metadata (SQLite/Peewee)
├── portal/
│   ├── app.py              # Flask Web Portal
│   ├── routes.py           # Ingest, Dashboard, Project Management
│   ├── sse_mon.py          # Flask-SSE relay for VRAM healthy
│   └── static/             # JS dashboard + SSE listeners
├── workers/
│   ├── base.py             # Base Class: cleanup(), release_lock()
│   ├── detection_worker.py # Whisper + PyAV (Audio cleaning suite)
│   ├── preview_worker.py   # ffmpegcv + RAMDisk (Sandbox engine)
│   └── render_worker.py    # ffmpegcv (Master rendering engine)
├── scripts/
│   └── setup_ramdisk.sh    # Script to init /dev/shm mount point
└── tests/
    └── integration/        # Hardware-bound lock testing
```

### Architectural Boundaries

**API Boundaries:**
All worker-to-portal communication is decoupled via Redis Streams. The Portal does not invoke workers directly (REST/RPC), ensuring that portal responsiveness is never blocked by VRAM intense processing.

**Component Boundaries:**
The **VRAM Guard** acts as the high-integrity boundary for all AI/Rendering tasks. No worker may initialize Torch or FFmpeg objects outside of a Guarded Context.

**Data Boundaries:**
- **Volatile**: `/dev/shm` is the boundary for low-latency sandbox previews.
- **Persistent**: Input/Output directories are the boundary for master renders and source files.

### Requirements to Structure Mapping

**Feature/Epic Mapping:**
- **Ingest & Management (FR1-3)**: `/portal` + `/core/db.py`
- **Intelligent Detection (FR4-7)**: `workers/detection_worker.py`
- **Verification & Sandbox (FR8-10)**: `workers/preview_worker.py` + `/dev/shm`
- **Hardware Governance (FR15-18)**: `core/vram_guard.py` + `portal/sse_mon.py`

### Integration Points

**Internal Communication:**
Uses the **Unified Redis Stream** pattern. `TaskDispatcher` XADDs to `jobs:stream`, and workers XREAD through consumer groups. Telemetry is piped from workers to `telemetry:stream` and relayed via Flask-SSE to the browser.

## Architecture Validation Results

### Coherence Validation ✅
- **Decision Compatibility**: Multi-process workers are perfectly decoupled via Redis Streams, preventing Flask blockages.
- **Pattern Consistency**: Telemetry streams are stateful with Redis history, supporting the "Always Consistent" dashboard requirement.
- **Structure Alignment**: Project layout separates core logic (`/core`) from transient worker execution (`/workers`), matching the VRAM Guard philosophy.

### Requirements Coverage Validation ✅
- **Functional Requirements**: 100% coverage. Every FR (from Ingest to Master Render) is mapped to a specific worker or core module.
- **Non-Functional Requirements**: GPU-acceleration (`ffmpegcv`) and RAMDisk usage directly address the <45s preview latency NFR.

### Implementation Readiness Validation ✅
- **Decision Completeness**: All critical technologies (FFmpeg, PyAV, Whisper, Redis Streams) are specified with roles.
- **Structure Completeness**: The directory tree provides clear homes for every module we’ve discussed.
- **Pattern Completeness**: OOM handling and VRAM safety buffers are enforced as "Mandatory for all agents."

### Architecture Completeness Checklist
- [x] Project context thoroughly analyzed
- [x] 12GB VRAM Scale and complexity assessed
- [x] Non-blocking SSE telemetry pipeline defined
- [x] Atomic Sequential Locking strategy secured
- [x] Complete directory structure defined
- [x] Component boundaries (VRAM/RAMDisk) established

### Architecture Readiness Assessment
**Overall Status**: READY FOR IMPLEMENTATION
**Confidence Level**: HIGH
**Key Strengths**: Aggressive VRAM management, zero-copy media optimization, and stateful telemetry.

### Implementation Handoff
**AI Agent Guidelines**: Use `pynvml` for VRAM guarding; NEVER initialize media workers outside of a Redis-locked context.
**First Implementation Priority**: `pip install ffmpegcv flask-sse pyav` followed by RAMDisk setup script development.
