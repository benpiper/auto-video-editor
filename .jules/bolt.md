## 2026-05-02 - Replace VideoFileClip with ffprobe for performance
**Learning:** `moviepy.editor.VideoFileClip` causes significant slowdowns for simple metadata (duration, resolution) and audio extraction on large videos, because it parses the whole file. Calling it purely for dimensions or duration takes ~150-200ms on a 5min video, and loading audio takes ~1-2.5s.
**Action:** Use `subprocess.run` with `ffprobe` to fetch metadata natively (`format=duration`, `stream=width,height`), and `ffmpeg` with `-acodec pcm_s16le` for much faster (and more reliable) audio extraction.
