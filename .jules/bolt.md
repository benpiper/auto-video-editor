
## 2026-05-29 - FFmpeg vs. PyDub Silence Detection
**Learning:** Loading entire media files into memory using `pydub`'s `AudioSegment` for minor metadata extraction (such as detecting silence intervals via `silence.detect_silence()`) is a huge performance bottleneck. `pydub` processes audio data entirely in Python loops and requires large memory buffers, taking around 8 seconds just to scan a 25-second audio file.
**Action:** Always favor native FFmpeg wrappers (`subprocess.run` calling `ffmpeg` with audio filters like `silencedetect`) over `pydub` or similar memory-heavy libraries. Native FFmpeg performs these operations in a highly optimized stream (e.g., dropping the time to ~0.26 seconds for the same task), conserving both memory and processing time.
