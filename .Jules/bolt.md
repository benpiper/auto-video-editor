## 2024-05-18 - Replacing moviepy with ffprobe/ffmpeg subprocess
**Learning:** Instantiating `VideoFileClip` from `moviepy.editor` introduces significant performance overhead when used solely to check if a video has an audio stream, or to read the duration and size of the video, as it initializes the full subclip and PyGame dependencies. By using raw `subprocess` calls to `ffprobe` (reading from `-of json`) and `ffmpeg`, we can gather properties and perform extractions in <100ms compared to `moviepy`'s >300ms overhead.
**Action:** When extracting audio, or querying video metadata like `duration` and `size`, favor `ffprobe -show_entries` rather than instantiating `moviepy` objects if `moviepy` is not otherwise needed. Be sure to handle cleanup of original code using `.close()` if replacing those initializations.
## 2025-01-25 - Caching Large ML Models
**Learning:** Large ML models (OpenAI Whisper and RVM) load from disk on every invocation, causing severe performance bottlenecks.
**Action:** Use `functools.lru_cache(maxsize=1)` to globally cache these models in memory. This prevents repeated disk loading while guarding against OOM issues by keeping the cache size limited.
