## 2026-05-11 - FFmpeg Subprocess vs MoviePy Overhead
**Learning:** Loading `moviepy.editor.VideoFileClip` incurs significant overhead (sometimes ~10x slower) just to read metadata or extract audio streams, due to Python object initialization and under-the-hood initialization logic.
**Action:** Always prefer direct `subprocess.run` calls to `ffprobe` for metadata fetching and `ffmpeg` for simple stream extraction over loading heavy wrapper objects like `VideoFileClip`.
