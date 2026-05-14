## 2024-05-14 - Bypass MoviePy overhead for simple audio/metadata extraction
**Learning:** Instantiating a `moviepy.editor.VideoFileClip` for simple tasks like metadata fetching (`duration`, `size`) or basic audio extraction introduces significant loading overhead (often >1s per instantiation) in this codebase's architecture.
**Action:** When extracting audio or fetching basic video dimensions/duration, prioritize direct `subprocess` calls using `ffmpeg` and `ffprobe` over loading full `VideoFileClip` objects to significantly speed up processing loops.
