## 2024-05-03 - Replaced moviepy with ffprobe for performance
**Learning:** `moviepy.editor.VideoFileClip` imposes an enormous constant overhead just to read video metadata (duration, width, height) or extract audio, simply because it builds extensive sub-objects and internal states (ffmpeg readers, audio readers).
**Action:** Always prefer native `subprocess` calls to `ffprobe` for grabbing metadata or `ffmpeg` for simple audio stream extraction to bypass library bloat.
