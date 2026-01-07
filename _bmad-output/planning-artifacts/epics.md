---
stepsCompleted: [1, 2, 3]
inputDocuments:
  - "/home/user/auto_video_editor/_bmad-output/planning-artifacts/prd.md"
  - "/home/user/auto_video_editor/_bmad-output/planning-artifacts/architecture.md"
  - "/home/user/auto_video_editor/_bmad-output/planning-artifacts/project-context.md"
---

# auto_video_editor - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for auto_video_editor, decomposing the requirements from the PRD, UX Design if it exists, and Architecture requirements into implementable stories.

## Requirements Inventory

### Functional Requirements

FR1: Users can upload high-resolution 4K video files for local processing.
FR2: Users can view and manage a list of current and completed video projects.
FR3: The system can track and report individual job progress across multiple worker stages.
FR4: The system can automatically detect periods of silence based on user-defined sensitivity.
FR5: The system can automatically detect verbatim filler words ("um," "uh") in the audio stream.
FR6: Users can independently enable or disable Silence and Filler detection for any given job.
FR7: Users can review a list of proposed segments to be removed before finalization.
FR8: Users can request a high-speed "Sandbox Preview" of the first 30 seconds of a project.
FR9: The system can render and display a preview clip that demonstrates the currently active cleaning settings.
FR10: Users can adjust cleaning sensitivity and trigger a fresh preview instantly.
FR11: Users can select between "Speed" (NVENC optimized) and "Quality" (high-fidelity) render presets.
FR12: The system can execute "Hard Cuts" to remove segments validated by the detection engine.
FR13: Users can initiate a "Master Render" of the full-length video once settings are validated.
FR14: Users can download or locally access the final exported .mp4 file.
FR15: The system can monitor real-time GPU VRAM availability on the host Linux machine.
FR16: Users can see a "Hardware Health" indicator within the web portal.
FR17: The system can alert the user if 12GB VRAM limits are exceeded or if CPU-fallback is detected.
FR18: The system can automatically queue and stagger AI vs. Rendering tasks to maintain OS stability.

### NonFunctional Requirements

NFR1: Draft Render Speed: For the "Speed" preset, the render-to-length ratio must remain < 0.5.
NFR2: Preview Latency: The "First-30s Sandbox" render must be completed and viewable in the web portal within 45 seconds of the request.
NFR3: Inference Accuracy: Verbatim transcription for filler word detection must achieve >90% accuracy on clear-audio inputs.
NFR4: VRAM Buffer: The system must maintain a minimum of 1.5GB free VRAM during all processing stages.
NFR5: Fault Isolation: AI Workers must be able to restart independently without affecting the Web Portal.
NFR6: Resource Staggering: Sequential task orchestration must prevent simultaneous peak memory usage by Whisper and NVENC.
NFR7: Local-Only Architecture: The entire pipeline must function with zero external internet dependency.
NFR8: Data Persistence: Temporary render files must be automatically purged after project completion.
NFR9: Real-time Monitoring: Hardware health metrics must refresh in the web portal interface every 2 seconds.
NFR10: Zero-Configuration: The tool should prioritize a "One-Click" cleaning experience optimized for 12GB NVIDIA hardware on Linux.

### Additional Requirements

- **VRAM Guard Strategy**: Atomic Sequential Locking via Redis. Only one high-memory worker (Whisper/NVENC) active at once.
- **Task Dispatch Pattern**: Unified Redis Streams (XADD/XREAD) for both tasking and telemetry.
- **Processing Storage**: High-Speed RAMDisk (`/dev/shm`) for all intermediate preview segments and isolated audio.
- **Telemetry State**: Stateful Redis Streams to support dashboard refresh and history playback.
- **Media Engine**: `ffmpegcv` for zero-copy GPU frame extraction and rendering.
- **Audio Prep**: `PyAV` for low-overhead audio stream demuxing before Whisper processing.
- **Worker Communication**: Redis Streams with Consumer Groups.
- **Telemetry Relay**: Flask-SSE relay for hardware health telemetry.
- **Environment Verification**: `workers/base.py` must verify `/dev/shm` capacity (>2GB free) before initialization.
- **Resiliency**: Circuit-breaker/retry decorator for Redis operations.
- **Naming Conventions**: `snake_case` (Python), hierarchical colon-separated (Redis), `kebab-case` (SSE).

### FR Coverage Map

FR1: Epic 2 - Media Ingest
FR2: Epic 2 - Project Management
FR3: Epic 2 - Job Progress Tracking
FR4: Epic 3 - Silence Detection
FR5: Epic 3 - Filler Word Detection
FR6: Epic 3 - Detection Configuration
FR7: Epic 3 - Removal Review List
FR8: Epic 4 - Sandbox Preview Request
FR9: Epic 4 - Preview Rendering
FR10: Epic 4 - Sensitivity Adjustment Loop
FR11: Epic 5 - Render Preset Selection
FR12: Epic 5 - Executive Removal (Hard Cuts)
FR13: Epic 5 - Master Render Initiation
FR14: Epic 5 - Export & Download
FR15: Epic 1 - Real-time VRAM Monitoring
FR16: Epic 1 - Hardware Health Dashboard UI
FR17: Epic 1 - Over-limit Threshold Alerts
FR18: Epic 1 - Task Staggering & Queue Management

## Epic List

## Epic 1: System Foundation & Hardware Governance

This epic establishes the high-integrity core of the application. It implements the "VRAM Guard" logic and the staggered Redis-backed task dispatcher to ensure the 12GB hardware limit is never breached during intensive media tasks.
**FRs covered:** FR15, FR16, FR17, FR18

### Story 1.1: Environment Initialization & RAMDisk Setup
As a system orchestrator, I want to automatically configure the Linux RAMDisk (/dev/shm) and verify its capacity, so that all volatile media processing stays on high-speed hardware with zero SSD wear.
**Acceptance Criteria:**
- **Given** a Linux host with root/sudo access for initial setup
- **When** the `scripts/setup_ramdisk.sh` is executed
- **Then** it must mount or verify a RAMDisk at `/dev/shm` with at least 4GB allocation
- **And** the `workers/base.py` `verify_environment()` method must hard-fail (SystemExit) if free space is `<2GB`.

### Story 1.2: Redis Client & Resilient Stream Wrappers
As a developer, I want a unified Redis client with built-in circuit-breaker and retry logic, so that inter-process communication remains stable during transient local network blips.
**Acceptance Criteria:**
- **Given** a local Redis server (v7.0+)
- **When** a worker or portal initializes the `core/redis_client.py`
- **Then** it must establish a connection pool with a circuit-breaker (50ms - 200ms threshold)
- **And** it must provide `add_to_stream` and `read_stream` wrappers.

### Story 1.3: The VRAM Guard (Atomic Sequential Locking)
As a system, I want to enforce a global hardware lock for high-VRAM tasks, so that only one resource-intensive worker (Whisper or NVENC) is active at a time.
**Acceptance Criteria:**
- **Given** multiple workers contending for the GPU
- **When** a worker attempts to enter a `with vram_guard():` context block
- **Then** it must successfully acquire the Redis key `lock:vram_atomic`
- **And** it must set a 60s inactivity timeout (TTL) to prevent system deadlocks.

### Story 1.4: Real-time VRAM Telemetry (pynvml)
As a system, I want to poll the actual hardware VRAM usage using NVIDIA's management library, so that the dashboard can display real-time health data.
**Acceptance Criteria:**
- **Given** a running `VRAM_Monitor` worker
- **When** it polls `pynvml` every 2 seconds
- **Then** it must XADD the `total`, `used`, and `free` bytes to the `telemetry:stream`.

### Story 1.5: Hardware Health SSE Relay
As a user, I want to see a live hardware health indicator on the web portal dashboard, so that I can verify my 12GB GPU is being utilized safely.
**Acceptance Criteria:**
- **Given** telemetry data arriving in the `telemetry:stream`
- **When** the Flask portal route `/sse/hardware` is requested
- **Then** it must use `Flask-SSE` to push new metrics to the browser every 2 seconds.

## Epic 2: Project Management & Media Ingest

Establishes the user interface and backend logic for handling 4K video uploads and project lifecycle management.
**FRs covered:** FR1, FR2, FR3

### Story 2.1: Local Project Database (SQLite/Peewee)
As a developer, I want a lightweight metadata store for project settings and job states, so that project data persists across restarts.
**Acceptance Criteria:**
- **Given** the application starts for the first time
- **When** the database initialization logic runs in `core/db.py`
- **Then** it must create a local SQLite database file `projects.db` with `Project` and `Task` models.

### Story 2.2: Fast Media Ingest & Metadata Extraction
As a user, I want to register local video files and see their metadata instantly, so that I can begin the cleaning process without manually entering specs.
**Acceptance Criteria:**
- **Given** a valid path to a 4K/1080p .mp4 or .mov file
- **When** I "Ingest" the file via the Flask portal route
- **Then** the system must use `PyAV` to extract duration, resolution, and audio/video stream parameters.

### Story 2.3: Project Management Dashboard
As an editor, I want a visual list of my video projects and their current status, so that I can manage my editing workload efficiently.
**Acceptance Criteria:**
- **Given** multiple projects in the database
- **When** I visit the `/dashboard` route
- **Then** the UI must list all projects with status badges (Ingested, Detecting, Ready).

### Story 2.4: Unified Job Progress Relay
As a user, I want to see real-time progress bars for background media tasks, so that I know exactly how long a detection or render will take.
**Acceptance Criteria:**
- **Given** a background worker is processing a task
- **When** the worker XADDs progress data to the Redis stream
- **Then** the Dashboard UI must update the corresponding progress bar in real-time via SSE.

### Story 2.5: Project Cleanup & Data Persistence Logic
As a user, I want the system to automatically purge temporary render files, so that my local storage doesn't get cluttered.
**Acceptance Criteria:**
- **Given** a project is marked as "Completed" or "Deleted"
- **When** the cleanup routine is triggered
- **Then** the system must delete all associated files in `/dev/shm` and temporary folders.

## Epic 3: Intelligence Suite (Detection Engine)

Integrates the AI models (Whisper/PyAV) and signal processing logic to identify silences and filler words within the media stream.
**FRs covered:** FR4, FR5, FR6, FR7

### Story 3.1: Precision Audio Extraction (PyAV)
As a system, I want to extract high-fidelity mono audio from video containers using zero-copy demuxing, so that AI models can process the audio with maximum accuracy.
**Acceptance Criteria:**
- **Given** a video ingest job
- **When** the `detection_worker.py` initiates audio extraction
- **Then** it must use `PyAV` to demux the primary audio stream to a 16kHz mono `.wav` file in `/dev/shm`.

### Story 3.2: Verbatim Filler Word Detection (Whisper + Regional Compile)
As an editor, I want the system to identify filler words like "um" and "uh" automatically, so that I don't have to manually scrub through the timeline.
**Acceptance Criteria:**
- **Given** an extracted audio file in `/dev/shm`
- **When** the Whisper model is invoked within a `vram_guard` context
- **Then** it must use `torch.compile` (Regional) to optimize inference speed and identify disfluencies with >90% precision.

### Story 3.3: Sensitivity-Aware Silence Detection
As an editor, I want to identify boring silences based on my own threshold settings, so that I can control the pacing of the edited video.
**Acceptance Criteria:**
- **Given** a mono audio file and a user-defined dB threshold
- **When** the silence detection logic runs
- **Then** it must identify all segments below the threshold for more than 2 seconds.

### Story 3.4: Detection Review API & Removal Candidate List
As a user, I want to see a list of all proposed cuts before the video is actually rendered, so that I can uncheck segments that I want to keep.
**Acceptance Criteria:**
- **Given** completed detection tasks
- **When** I request the "Review List" via the web portal
- **Then** the UI must display a list of all candidates with `type`, `timestamp`, and an `included` checkbox.

## Epic 4: The "30s Sandbox" Preview Engine

Implements the high-speed preview engine using /dev/shm for low-latency feedback on cleaning settings.
**FRs covered:** FR8, FR9, FR10

### Story 4.1: Sandbox Segment Extractor (ffmpegcv)
As a system, I want to isolate the first 30 seconds of high-resolution video using zero-copy extraction, so that I can provide a representative sample for preview.
**Acceptance Criteria:**
- **Given** a project with an ingested 4K video
- **When** a Sandbox Preview is requested
- **Then** the system must use `ffmpegcv` to extract the first 30 seconds to `/dev/shm` in `< 5 seconds`.

### Story 4.2: Dynamic Preview Assembler (Hardware-Accelerated)
As an editor, I want to see my cleaning settings applied to the preview segment instantly, so that I can verify my choices before starting the final render.
**Acceptance Criteria:**
- **Given** an extracted 30s segment and a list of validated removal segments
- **When** the "Render Preview" task is dispatched
- **Then** the `preview_worker.py` must use `ffmpegcv` and NVENC to execute hard cuts in `< 45 seconds`.

### Story 4.3: Instant Sensitivity Feedback Loop (Portal Integration)
As a user, I want to adjust my detection settings and see an updated preview without a full page refresh, so that I can rapidly iterate.
**Acceptance Criteria:**
- **Given** an active Project Preview page
- **When** I change a threshold and click "Re-Preview"
- **Then** the UI must trigger a fresh assembly job and update the `<video>` player dynamically.

## Epic 5: Master Rendering & Quality Presets

The final production stage where the full-length render is executed using hardware-accelerated codecs (NVENC).
**FRs covered:** FR11, FR12, FR13, FR14

### Story 5.1: "Speed" Preset (NVENC Single-Pass Hard Cut)
As a procrastinating reviewer, I want to render my cleaned rough-cut as fast as possible, so that I can immediately move to my editorial stage.
**Acceptance Criteria:**
- **Given** a project with validated removal segments
- **When** I select the "Speed" preset
- **Then** the system must use `ffmpegcv` with `h264_nvenc` and execute hard cuts with a render ratio `< 0.5`.

### Story 5.2: "Quality" Preset (High-Fidelity Master)
As a high-stakes presenter, I want to render my cleaned video at the highest possible quality, so that the automated edits are visually indistinguishable from source.
**Acceptance Criteria:**
- **Given** a project with reviewed segments
- **When** I select the "Quality" preset
- **Then** the system must use `ffmpegcv` with `hevc_nvenc` (p7 preset) and maintain source resolution.

### Story 5.3: Master Render Orchestrator
As a system, I want to manage the full-length render process with strict VRAM governance, so that the system remains stable.
**Acceptance Criteria:**
- **Given** a master render job
- **When** the `render_worker.py` begins processing
- **Then** it must hold the `lock:vram_atomic` and provide 1% incremental progress telemetry.

### Story 5.4: Export Manager & Post-Render Notifications
As a user, I want to be notified when my master render is ready and easily access the file, so that I can move on to my next task.
**Acceptance Criteria:**
- **Given** a completed master render
- **When** the worker signals task completion
- **Then** the web portal must display a success notification and provide a "Download" button.
