## 2024-05-12 - Video Metadata and Audio Extraction Overhead
**Learning:** Using `moviepy.editor.VideoFileClip` just to fetch duration, size, or extract audio is extremely inefficient because it loads video decoding pipelines (FFMPEG wrapper + imageio) entirely into memory before starting. It is an order of magnitude slower than native ffprobe/ffmpeg subprocess calls.
**Action:** Replace `VideoFileClip` metadata lookups (`video.duration`, `video.size`) with `ffprobe`, and `video.audio.write_audiofile()` with `ffmpeg` subprocess calls when moviepy decoding features are not strictly required.
