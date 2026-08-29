## 2024-05-18 - Replacing moviepy with ffprobe/ffmpeg subprocess
**Learning:** Instantiating `VideoFileClip` from `moviepy.editor` introduces significant performance overhead when used solely to check if a video has an audio stream, or to read the duration and size of the video, as it initializes the full subclip and PyGame dependencies. By using raw `subprocess` calls to `ffprobe` (reading from `-of json`) and `ffmpeg`, we can gather properties and perform extractions in <100ms compared to `moviepy`'s >300ms overhead.
**Action:** When extracting audio, or querying video metadata like `duration` and `size`, favor `ffprobe -show_entries` rather than instantiating `moviepy` objects if `moviepy` is not otherwise needed. Be sure to handle cleanup of original code using `.close()` if replacing those initializations.

## 2024-08-29 - Globally caching large ML models
**Learning:** Instantiating large ML models (like Whisper and RVM) repeatedly causes severe performance bottlenecks from disk loading and risks Out-Of-Memory (OOM) issues due to memory leaks.
**Action:** Always globally cache large models using `@functools.lru_cache(maxsize=1)` to ensure they are loaded into memory exactly once per process.
