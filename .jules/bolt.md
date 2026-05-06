## 2024-05-06 - Initial

## 2024-05-06 - Replace moviepy.editor with ffprobe/ffmpeg in processor.py
**Learning:** Loading video using `moviepy.editor.VideoFileClip` is significantly slower than executing native `ffprobe` and `ffmpeg` commands via `subprocess`, especially when we only need to extract audio or read basic metadata like duration and size. This is due to Python loading a significant amount of overhead that moviepy requires to create clip instances.
**Action:** Avoid using `moviepy.editor.VideoFileClip` when only extracting audio streams or fetching metadata (like size or duration). Prefer native `ffprobe` and `ffmpeg` `subprocess` calls.
