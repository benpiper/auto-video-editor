## 2026-05-10 - Avoiding `moviepy.editor.VideoFileClip` for simple data
**Learning:** `moviepy.editor.VideoFileClip` initializes a full FFmpeg background process, parses stdout frame-by-frame, and constructs large internal objects in memory even just to get duration/size or extract audio.
**Action:** Use native `subprocess.run(['ffprobe', ...])` or `subprocess.run(['ffmpeg', ...])` for lightweight metadata/audio extraction. This provides huge speedups (e.g. 5.0s to 4.1s for extracting audio, and 0.16s to 0.08s for basic metadata reading).
