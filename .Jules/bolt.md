## 2024-05-18 - Replacing moviepy with ffprobe/ffmpeg subprocess
**Learning:** Instantiating `VideoFileClip` from `moviepy.editor` introduces significant performance overhead when used solely to check if a video has an audio stream, or to read the duration and size of the video, as it initializes the full subclip and PyGame dependencies. By using raw `subprocess` calls to `ffprobe` (reading from `-of json`) and `ffmpeg`, we can gather properties and perform extractions in <100ms compared to `moviepy`'s >300ms overhead.

## 2026-08-17 - Cache Large ML Models to Prevent Disk I/O Bottlenecks
**Learning:** Loading large ML models (like Whisper and RVM) directly from disk on every invocation causes severe performance bottlenecks due to repeated I/O operations and initialization overhead.
**Action:** Always wrap model loading functions with `@functools.lru_cache(maxsize=1)` to globally cache the models in memory, ensuring subsequent calls are instantaneous while preventing Out-Of-Memory (OOM) issues by strictly limiting the cache size.
