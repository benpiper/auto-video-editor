## 2026-05-21 - Use Subprocesses for Media Analysis
**Learning:** Loading entire media files into memory using libraries like `pydub.AudioSegment` for simple metadata analysis or silence detection creates massive overhead.
**Action:** Always prefer native `ffprobe` or `ffmpeg` streamed analysis via `subprocess` with `-f null -` for metadata and silence detection to keep the footprint O(1).
