# Story 3.1: Precision Audio Extraction (PyAV)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a system,
I want to extract high-fidelity mono audio from video containers using zero-copy demuxing,
so that AI models can process the audio with maximum accuracy.

## Acceptance Criteria

1. **Given** a video project, **When** the detection pipeline starts, **Then** the system must use `PyAV` to extract the primary audio stream.
2. **Given** an extraction task, **When** completed, **Then** the resulting file must be a 16kHz mono `.wav` file.
3. **Given** high-performance requirements, **When** extraction occurs, **Then** it should leverage the RAMDisk (`/dev/shm`) for temporary storage of the audio file to minimize Disk I/O.
4. **Given** the worker architecture, **When** extraction is happening, **Then** progress must be reported to the `progress:stream` in Redis.

## Tasks / Subtasks

- [x] Define `AUDIO_EXTRACT_PATH` typically in `/dev/shm` or configured temp dir (AC: 3)
- [x] Implement `core/audio_extract.py` utility using PyAV (AC: 1, 2)
  - [x] Support custom sampling rate (default 16000)
  - [x] Support mono downmixing
- [x] Integrate with `processor.py` (replacing the current MoviePy extract if applicable) (AC: 1)
- [x] Add progress reporting during extraction (AC: 4)
- [x] Implement unit tests for audio extraction
  - [x] Verify file format (16kHz, mono, wav)
  - [x] Verify performance (speed)

## Dev Notes

- **PyAV Efficiency**: Demuxing via PyAV is significantly faster than MoviePy's ffmpeg-subprocess approach.
- **RAMDisk Integration**: Automatically detects and uses `/dev/shm` for temporary audio storage, drastically reducing Disk I/O wait times.
- **Whisper Ready**: Output is hard-coded to 16kHz mono pcm_s16le, the optimal format for Faster-Whisper and OpenAI Whisper.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 3.1]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Pytest run `tests/test_audio_extract.py`: 3 passed.
- Fixed `av` Layout issue (setting `.layout` on codec context).

### Completion Notes List

- Created `core/audio_extract.py`.
- Updated `processor.py` to use PyAV and RAMDisk.
- Verified progress mapping from sub-task to main job (5-10% of total job).

### File List
- `core/audio_extract.py`
- `processor.py`
- `tests/test_audio_extract.py`
