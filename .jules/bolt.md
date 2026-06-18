## 2024-05-24 - Prefer FFmpeg over PyDub for Audio Analysis
**Learning:** Using `pydub.silence.detect_silence` loads the entire audio file into memory as an `AudioSegment` and iterates through it, causing massive overhead and slow processing times.
**Action:** Use native FFmpeg capabilities via `subprocess` (e.g., `silencedetect` filter) for simple audio analysis and metadata fetching to significantly reduce processing overhead and memory usage.
