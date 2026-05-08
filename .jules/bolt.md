## 2024-05-08 - FFprobe/FFmpeg native calls over MoviePy VideoFileClip
**Learning:** Loading a `VideoFileClip` via MoviePy incurs significant overhead even for simple metadata fetching or audio extraction operations because it initializes the full reader backend.
**Action:** Always prefer native `ffprobe` (to fetch JSON metadata like duration, dimensions) and `ffmpeg` (via `subprocess.run`) for lightweight pre-processing operations instead of initializing heavy MoviePy objects.
