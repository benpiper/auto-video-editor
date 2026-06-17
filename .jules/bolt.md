## 2024-05-24 - Model Loading Bottleneck
**Learning:** Large ML models like OpenAI Whisper and RVM are loaded from disk every time they are called, causing a massive I/O and processing bottleneck, especially when processing multiple jobs.
**Action:** Use `@functools.lru_cache(maxsize=1)` on model loading functions to keep them in memory for subsequent calls without causing OOM issues.
