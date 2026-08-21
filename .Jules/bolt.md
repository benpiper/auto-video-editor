## 2024-05-18 - Replacing moviepy with ffprobe/ffmpeg subprocess
**Learning:** Instantiating `VideoFileClip` from `moviepy.editor` introduces significant performance overhead when used solely to check if a video has an audio stream, or to read the duration and size of the video, as it initializes the full subclip and PyGame dependencies. By using raw `subprocess` calls to `ffprobe` (reading from `-of json`) and `ffmpeg`, we can gather properties and perform extractions in <100ms compared to `moviepy`'s >300ms overhead.
**Action:** When extracting audio, or querying video metadata like `duration` and `size`, favor `ffprobe -show_entries` rather than instantiating `moviepy` objects if `moviepy` is not otherwise needed. Be sure to handle cleanup of original code using `.close()` if replacing those initializations.

## 2026-08-21 - Caching ML Models in Memory
**Learning:** Large ML models like OpenAI Whisper (in `processor.py`) and RVM (in `background_remover.py`) have a significant initialization overhead. Loading them from disk on every invocation introduces a severe performance bottleneck.
**Action:** Use `@functools.lru_cache(maxsize=1)` to cache the instantiated ML model in memory globally. This prevents repeated disk loading during subsequent processing runs, significantly improving performance while avoiding Out-Of-Memory (OOM) issues.
