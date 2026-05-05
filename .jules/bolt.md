## 2025-02-24 - Faster metadata extraction
**Learning:** `moviepy.editor.VideoFileClip` loads the entire video configuration, parses multiple streams and builds complex python objects which takes over 1.1s even for a simple video metadata read. Extracting audio using `write_audiofile()` is also slower than calling ffmpeg directly.
**Action:** When extracting audio or just grabbing duration/size, use raw `subprocess` calls to `ffmpeg` and `ffprobe` rather than loading a `VideoFileClip`.
