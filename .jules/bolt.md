## 2025-02-19 - Replace pydub with native ffmpeg for silence detection
**Learning:** Loading large video/audio files into Python memory using libraries like `pydub` just to fetch simple metadata or detect silence scales very poorly and creates a massive bottleneck.
**Action:** When working with video/audio pipelines, favor native `ffmpeg`/`ffprobe` subprocess calls over Python wrapper libraries for large file scans, avoiding loading the entire file into memory.
