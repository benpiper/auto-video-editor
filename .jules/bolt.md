## 2026-05-17 - MoviePy Overhead for Metadata and Audio Extraction
**Learning:** Using `moviepy.editor.VideoFileClip` just to fetch video metadata (duration, size) or extract the audio track introduces severe performance overhead. It initializes an entire video processing pipeline when it's not needed, and the import overhead of MoviePy itself is high.
**Action:** When basic video metadata or audio extraction is required, always prefer native `subprocess.run` calls to `ffprobe` and `ffmpeg`. Reserve `moviepy` only for complex editing/concatenation steps.
