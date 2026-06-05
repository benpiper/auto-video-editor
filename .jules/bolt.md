## 2026-06-05 - Global ML Model Caching
**Learning:** Large ML models (e.g., OpenAI Whisper and Robust Video Matting) cause severe performance bottlenecks when loaded from disk repeatedly on each function call during video processing.
**Action:** Use `functools.lru_cache(maxsize=1)` to globally cache these models in memory, preventing redundant disk I/O while guarding against potential Out-Of-Memory (OOM) issues compared to standard dictionary caches.
