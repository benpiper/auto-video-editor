## 2024-05-20 - Fast Audio Analysis via Native Filters
**Learning:** Using native FFmpeg filters via `subprocess` (like `silencedetect`) is vastly faster than Python libraries that load entire media files into memory for analysis (like `pydub` or `moviepy`). `pydub` `detect_silence` took ~3.6s vs FFmpeg's `silencedetect` which took ~0.16s, a 22x speedup.
**Action:** When analyzing media files for specific properties (silence, volume, metadata), always check if FFmpeg/FFprobe has a native filter for it before resorting to Python libraries that load the media array into memory.
