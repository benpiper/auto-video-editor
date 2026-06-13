## 2026-06-13 - Cache ML models to prevent repeated loading
**Learning:** Large ML models like Whisper and RVM take significant time to load from disk on each request, causing severe performance bottlenecks.
**Action:** Use `functools.lru_cache(maxsize=1)` to cache loaded ML models in memory to avoid repeated disk I/O, preventing Out-Of-Memory (OOM) issues while improving processing speed.
