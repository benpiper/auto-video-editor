## 2024-05-18 - Replacing moviepy with ffprobe/ffmpeg subprocess
**Learning:** Instantiating `VideoFileClip` from `moviepy.editor` introduces significant performance overhead when used solely to check if a video has an audio stream, or to read the duration and size of the video, as it initializes the full subclip and PyGame dependencies. By using raw `subprocess` calls to `ffprobe` (reading from `-of json`) and `ffmpeg`, we can gather properties and perform extractions in <100ms compared to `moviepy`'s >300ms overhead.
**Action:** When extracting audio, or querying video metadata like `duration` and `size`, favor `ffprobe -show_entries` rather than instantiating `moviepy` objects if `moviepy` is not otherwise needed. Be sure to handle cleanup of original code using `.close()` if replacing those initializations.

## 2026-09-01 - Cache ML models in memory
**Learning:** Large ML models like Whisper and RVM cause severe performance bottlenecks and OOM issues due to repeated disk loading during processing if they are instantiated multiple times or repeatedly in different scopes.
**Action:** Use `functools.lru_cache(maxsize=1)` to globally cache these models in memory. This guards against OOM issues by ensuring at most one model instance per config is kept in memory while eliminating redundant loading overhead.
