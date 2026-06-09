## 2024-05-15 - Model Loading Bottleneck
**Learning:** Large ML models (like Whisper and RVM) are repeatedly loaded from disk during each processing run, causing severe I/O bottlenecks and increased processing time.
**Action:** Use `functools.lru_cache(maxsize=1)` on model loading functions to keep them in memory across runs, significantly speeding up subsequent requests while preventing OOM errors.
