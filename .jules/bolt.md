## 2026-09-11 - Globally caching large ML models
**Learning:** Large models like Whisper and RVM cause severe performance bottlenecks and potential Out-Of-Memory (OOM) issues if loaded repeatedly from disk.
**Action:** Use `functools.lru_cache(maxsize=1)` to globally cache these large ML models in memory to avoid repeated disk loading.
