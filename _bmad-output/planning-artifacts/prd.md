---
stepsCompleted: [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
inputDocuments:
  - "/home/user/auto_video_editor/_bmad-output/planning-artifacts/research/technical-Optimizing-Automated-Video-Processing-Pipelines-research-2026-01-05.md"
  - "/home/user/auto_video_editor/docs/api-contracts.md"
  - "/home/user/auto_video_editor/docs/architecture-core.md"
  - "/home/user/auto_video_editor/docs/architecture-web_portal.md"
  - "/home/user/auto_video_editor/docs/data-models.md"
  - "/home/user/auto_video_editor/docs/development-guide.md"
  - "/home/user/auto_video_editor/docs/index.md"
  - "/home/user/auto_video_editor/docs/integration-architecture.md"
  - "/home/user/auto_video_editor/docs/project-overview.md"
  - "/home/user/auto_video_editor/docs/project-scan-report.json"
  - "/home/user/auto_video_editor/docs/source-tree-analysis.md"
  - "/home/user/auto_video_editor/docs/ui-component-inventory.md"
workflowType: 'prd'
lastStep: 0
briefCount: 0
researchCount: 1
brainstormingCount: 0
projectDocsCount: 11
---

# Product Requirements Document - auto_video_editor

**Author:** Ben
**Date:** 2026-01-05

## Executive Summary

The `auto_video_editor` is being transformed from a linear utility into a high-performance, automated media production engine. This update introduces an intelligent "Cleaning Suite" that allows users to automatically strip disfluencies (um, uh), silence, and visual "still segments" based on their specific needs. By moving to a hardware-accelerated, staggered pipeline, the system maximizes a single 12GB GPU to deliver professional results without the need for cloud-based processing.

### What Makes This Special

- **Granular Removal Toggle**: Unlike "black-box" slicers, this system lets the user decide exactly what is cut. You can remove filler words while keeping pauses for dramatic effect, or strip still segments while keeping the audio intact.
- **Dynamic Resource Presets**: A Speed/Quality toggle solves the video encoding bottleneck. Users can choose "Speed" for quick drafts using fast hardware presets, or "Quality" for high-bitrate final exports that preserve every detail.
- **Professional Polish (Fades)**: Users can optionally select cross-fade transitions instead of hard cuts. While this increases processing time, it significantly improves the production value of the final output.
- **Hardware Health Monitor**: A novel "VRAM Guard" that detects in real-time if the 12GB budget is exceeded. It alerts the user if the system has silently switched to slow CPU processing, ensuring the "performance cliff" is transparently managed.

## Project Classification

**Technical Type:** `web_app` / `api_backend` (Media Processing Engine)
**Domain:** `scientific` (AI/ML Media Pipelines)
**Complexity:** `medium`
**Project Context:** Brownfield - extending existing system

The project is a performance-heavy media processing application. It requires deep integration between high-level AI inference (Whisper/Torch) and low-level hardware codecs (NVENC/NVDEC). The complexity is rated as **Medium** because while the core models exist, the orchestration of a staggered, memory-aware pipeline on a fixed 12GB VRAM budget requires precision.

## Success Criteria

### User Success

- **80% Manual Labor Reduction**: Success is defined as a user being able to perform a complete "rough cut" (removing silence, disfluencies, and still segments) in 20% of the time it previously took to do manually.
- **Cognitive Ease**: The "Aha!" moment occurs when the editor presents a fully populated timeline of suggested cuts within 5 minutes of a raw video upload.
- **Production Confidence**: Users feel empowered to use the "Quality" preset for final exports, trusting that the automated fades and hardware-accelerated render will be indistinguishable from a pro-level manual edit.

### Business & Productivity Success

- **Daily Driver Status**: The tool becomes the primary entry point for all 4K video editing, replacing manual scrubbing for 95% of initial filler content detection.
- **Infrastructure ROI**: Achieving cloud-level processing speeds (10m video in < 3m) on a local 12GB card, validating the local-first architecture.

### Technical Success

- **Rock-Solid Reliability**: 100% stable execution during concurrent Matting + Transcription tasks. The system must never crash due to VRAM exhaustion (OOM), instead utilizing the "VRAM Guard" to throttle or alert.
- **Hardware Accuracy**: >90% precision in detecting "Stills" and "Filler Words" using the existing Whisper model and NVENC/NVDEC hardware blocks.
- **Transparency**: The "Hardware Health Monitor" accurately identifies and reports any silent CPU-fallback within 2 seconds of a performance drop.

## Product Scope

### MVP - Minimum Viable Product

- **Triple-Threat Detection**: Automated detection and removal of Silence, Filler Words (um, uh), and Still segments.
- **User Agency Toggles**: Checkbox-based selection to enable/disable specific removal types (e.g., "Keep stills but remove fillers").
- **Speed vs. Quality Presets**: 
    - **Speed**: Single-pass NVENC hardware encode with hard cuts.
    - **Quality**: Multi-pass high-fidelity encode with professional cross-fade transitions.
- **VRAM Guard & Monitor**: Real-time hardware health monitor to ensure 12GB stability on Linux.
- **Staggered Pipeline**: Smart worker orchestration to manage VRAM peak usage.

### Growth Features (Post-MVP)

- **Custom Cross-Fade Timing**: User-definable durations for transition effects.
- **Perceptual Audio Cleanup**: Advanced filtering to remove background hum or mouse clicks alongside filler words.
- **Multi-Track Preview**: Side-by-side comparison of "Raw" vs "Cleaned" versions before export.

### Vision (Future)

- **Autonomous "Best Take" Selection**: Identifying and choosing the best version of a repeated line based on audio clarity and visual stillness.
- **Context-Aware B-Roll**: Automatic suggestions for B-roll overlays based on the transcription text.

## User Journeys

**Journey 1: Alex - The "Procrastinating Product Reviewer" (Efficiency Path)**
Alex has 45 minutes of raw 4K footage from a tech review. It's full of "umms," long pauses, and segments where he's just staring at the camera (stills). He needs a rough cut *now* so he can start adding B-roll. He opens `auto_video_editor`, selects all cleaning toggles, and hits the "Speed" preset. Within minutes, the 45-minute slog is a 12-minute clean sequence. He's already adding overlays while his competitors are still on their first cup of coffee.

**Journey 2: Sarah - The "Tech-Savvy Operations Manager" (Troubleshooting Path)**
Sarah is processing a massive high-resolution webinar on her 12GB Linux workstation while also running a heavy browser-based research session. She starts the "Quality" render. Midway through, the **Hardware Health Monitor** pings her: "VRAM budget exceeded; potential CPU-fallback detected." She quickly closes her extra browser tabs, and the monitor updates: "GPU processing restored." Sarah breathes a sigh of relief—the 12GB limit was guarded, and her render didn't turn into a multi-hour nightmare.

**Journey 3: Marcus - The "High-Stakes Presenter" (Quality Path)**
Marcus is creating an internal company video that needs to look polished. He isn't in a rush, but the transitions must be seamless. He enables "Filler Word Removal" and selects the **"Quality" Preset with Cross-Fades**. Even though the rendering takes 3x longer than the hard-cut version, the resulting video feels fluid and professional. The "ums" are gone, but more importantly, the edits are invisible. Marcus presents a video that looks like it was edited by a pro agency, all from his local machine.

### Journey Requirements Summary

These journeys reveal several critical capability areas:
- **Detection UI**: Toggles for specific removal types (Silence, Filler, Still).
- **Preset Management**: Logic to switch between NVENC speed profiles and high-fidelity multi-pass filters.
- **Hardware Overlay**: A real-time status indicator or notification system for VRAM health.
- **Transition Engine**: Support for both "Fast-Cut" (Stream Copy) and "Smooth-Fade" (Re-encode) operations.

## Innovation & Novel Patterns

### Detected Innovation Areas
- **Hardware-Aware "VRAM Guarding"**: A first-principles approach to local AI stability. Instead of letting the OS handle memory overflow (which leads to the 90% performance cliff of CPU-fallback), the system proactively monitors and manages the 12GB budget.
- **The "Sandbox" Preview Logic**: A novel validation loop that allows users to instantly preview "Cleaning Suite" settings on high-fidelity segments before committing the hardware to a multi-hour 4K render.
- **Unified Cleaning Pass**: Rethinking the sequential "edit then clean" workflow. This innovation combines three distinct AI detection tasks (Silence, Filler, Still) into a single hardware-optimized pipeline.

### Market Context & Competitive Landscape
- **Local vs. SaaS**: We are competing with cloud-based tools like Descript or Riverside. Our innovation is providing that same "Magic" but with zero monthly cost, total privacy, and direct hardware utilization.

### Validation Approach
- **The A/B Toggle**: The preview feature will be the primary validation tool, allowing users to rapidly A/B test "Hard Cuts" vs "Smooth Fades" in real-time.

### Risk Mitigation
- **Fallback Transparency**: If the "Sandbox Preview" performs slowly, the system uses its internal health monitor to warn the user *before* they start the main job.

## Media Processing Specific Requirements

### Project-Type Overview
The `auto_video_editor` is a hybrid Web Application and Worker-based Media Engine. It utilizes a Flask-based management portal to orchestrate specialized Python workers for AI inference (Whisper/Torch) and hardware-accelerated video processing (FFmpeg/NVENC).

### Technical Architecture Considerations
- **Internal Worker API**: The system must maintain a stable internal API between the Web Portal and the Redis-backed workers to handle state transitions for the "Cleaning Suite" (Detection → Cleaning → Render).
- **Segmented Rendering**: The "Sandbox Preview" logic will utilize temporary file storage to render 30-second clips at the user-selected quality preset, allowing for non-destructive testing of parameters.
- **Hardware-Aware Throttling**: The "VRAM Guard" logic will implement a priority-queue in the task worker. If VRAM availability drops below a critical threshold (e.g., < 1GB free), the system will automatically pause non-essential workers or reduce the number of parallel FFmpeg streams to prevent OS-level lag or CPU-fallback.

### Data & Performance Requirements
- **Format Support Matrix**: 
  - **Inputs**: Native 4K/1080p support for `.mp4` (H.264/H.265) and `.mov`.
  - **Outputs**: Browser-compatible `.mp4` for web previews; user-defined quality for final exports.
- **Performance Targets**: 
  - **Inference Latency**: Total detection time (Silence + Filler + Still) should not exceed 50% of real-time video duration.
  - **Preview Generation**: 30-second rendered clip must be available in the portal within 45 seconds of request.

### Browser & UI Requirements
- **Modern Browser Support**: Optimized for Chrome/Firefox on Linux (X11/Wayland) for the management dashboard.
- **Local Dev Server**: The portal runs as a local Flask instance (`localhost`), prioritizing low latency and local file-system access over remote accessibility.

### Implementation Considerations
- **Non-Invasive Architecture**: The update must maintain compatibility with current Dockerized worker patterns.
- **Zero-Copy Optimization**: Implementation should prioritize `ffmpegcv` and shared memory buffers wherever possible to minimize the impact on the 12GB VRAM budget.

## Project Scoping & Phased Development

### MVP Strategy & Philosophy
**MVP Approach:** Experience-First MVP. We are prioritizing the "Sandbox Preview" feedback loop for the first 30 seconds of video to ensure settings are correct before a full render.
**Resource Requirements:** Solo-Dev with Python/FFmpeg/Flask expertise. 12GB NVIDIA GPU required for validation.

### MVP Feature Set (Phase 1)
**Core User Journeys Supported:**
- **Journey 1 (Efficiency)**: Rapid 4K cleaning.
- **Journey 2 (Troubleshooting)**: Hardware health monitoring.

**Must-Have Capabilities:**
- **The "First-30s" Sandbox**: Instant Head-of-Clip preview logic.
- **Audio Cleaning Suite**: Silence Removal & Filler Word Detection (Um/Uh).
- **Quality Presets**: "Speed" (NVENC hard cuts) vs "Quality" (high-fidelity).
- **Hardware Stability**: VRAM Guard (Alert only) and sequential task orchestration.

### Post-MVP Features
**Phase 2 (Growth):**
- **Still Segment Detection**: Visual detection of static segments.
- **Middle-of-Clip Previews**: Arbitrary seek-time for sandbox testing.
- **Professional Fades**: Cross-fade transitions (re-rendering).
- **Automated VRAM Throttling**: Active worker adjustment.

**Phase 3 (Expansion):**
- **Cadence Learning**: AI personalized to user speech styles.
- **Interactive Timeline**: Pre-render visualization of cuts.

### Risk Mitigation Strategy
- **Technical**: Preview latency. Mitigation: Forced high-speed NVENC for previews.
- **Hardware**: 12GB peak usage. Mitigation: Moving Still Detection to Phase 2 and enforcing sequential worker execution.
- **UX**: User frustration with incorrect detection. Mitigation: First-30s preview allows rapid threshold tuning.

## Functional Requirements

### Ingest & Project Management (Leveraging Existing Infrastructure)
- **FR1**: Users can upload high-resolution 4K video files for local processing.
- **FR2**: Users can view and manage a list of current and completed video projects.
- **FR3**: The system can track and report individual job progress across multiple worker stages.

### Intelligent Detection (The Cleaning Suite)
- **FR4**: The system can automatically detect periods of silence based on user-defined sensitivity.
- **FR5**: The system can automatically detect verbatim filler words ("um," "uh") in the audio stream.
- **FR6**: Users can independently enable or disable Silence and Filler detection for any given job.
- **FR7**: Users can review a list of proposed segments to be removed before finalization.

### Verification & The Sandbox (MVP Focus)
- **FR8**: Users can request a high-speed "Sandbox Preview" of the first 30 seconds of a project.
- **FR9**: The system can render and display a preview clip that demonstrates the currently active cleaning settings.
- **FR10**: Users can adjust cleaning sensitivity and trigger a fresh preview instantly.

### Post-Processing & Master Rendering
- **FR11**: Users can select between "Speed" (NVENC optimized) and "Quality" (high-fidelity) render presets.
- **FR12**: The system can execute "Hard Cuts" to remove segments validated by the detection engine.
- **FR13**: Users can initiate a "Master Render" of the full-length video once settings are validated.
- **FR14**: Users can download or locally access the final exported .mp4 file.

### Hardware & System Governance (The VRAM Guard)
- **FR15**: The system can monitor real-time GPU VRAM availability on the host Linux machine.
- **FR16**: Users can see a "Hardware Health" indicator within the web portal.
- **FR17**: The system can alert the user if 12GB VRAM limits are exceeded or if CPU-fallback is detected.
- **FR18**: The system can automatically queue and stagger AI vs. Rendering tasks to maintain OS stability.

## Non-Functional Requirements

### Performance
- **Draft Render Speed**: For the "Speed" preset, the render-to-length ratio must remain < 0.5 (e.g., 10m video cleans in < 5m).
- **Preview Latency**: The "First-30s Sandbox" render must be completed and viewable in the web portal within 45 seconds of the request.
- **Inference Accuracy**: Verbatim transcription for filler word detection must achieve >90% accuracy on clear-audio inputs.

### Reliability & Stability (Hardware Governance)
- **VRAM Buffer**: The system must maintain a minimum of 1.5GB free VRAM during all processing stages to ensure Linux desktop stability.
- **Fault Isolation**: AI Workers must be able to restart independently without affecting the Web Portal or other pending jobs.
- **Resource Staggering**: Sequential task orchestration must prevent simultaneous peak memory usage by Whisper and NVENC.

### Security & Privacy
- **Local-Only Architecture**: The entire pipeline (Detection, Transcription, Rendering) must function with zero external internet dependency to ensure total content privacy.
- **Data Persistence**: Temporary render files must be automatically purged after project completion or via explicit user request.

### Usability
- **Real-time Monitoring**: Hardware health metrics (VRAM, CPU, Task Status) must refresh in the web portal interface every 2 seconds during active processing.
- **Zero-Configuration**: The tool should prioritize a "One-Click" cleaning experience optimized for 12GB NVIDIA hardware on Linux.
