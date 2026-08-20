## 2024-05-18 - Replacing moviepy with ffprobe/ffmpeg subprocess
**Learning:** Instantiating `VideoFileClip` from `moviepy.editor` introduces significant performance overhead when used solely to check if a video has an audio stream, or to read the duration and size of the video, as it initializes the full subclip and PyGame dependencies. By using raw `subprocess` calls to `ffprobe` (reading from `-of json`) and `ffmpeg`, we can gather properties and perform extractions in <100ms compared to `moviepy`'s >300ms overhead.
**Action:** When extracting audio, or querying video metadata like `duration` and `size`, favor `ffprobe -show_entries` rather than instantiating `moviepy` objects if `moviepy` is not otherwise needed. Be sure to handle cleanup of original code using `.close()` if replacing those initializations.

## 2026-08-20 - Global ML Model Caching in Memory
**Learning:** Repeatedly loading large ML models (like OpenAI Whisper or RVM) from disk for every processing request creates a severe performance bottleneck and potential Out-Of-Memory (OOM) issues in a server environment.
**Action:** Always use `functools.lru_cache(maxsize=1)` on model loading functions to globally cache ML models in memory across requests.
