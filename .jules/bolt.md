## 2026-09-10 - Prevent OOM and disk bottleneck via ML Model Caching
**Learning:** Large ML models like OpenAI Whisper and RVM cause severe performance bottlenecks from repeated disk loading and can lead to Out-Of-Memory (OOM) issues if loaded multiple times during repeated processing tasks.
**Action:** Always globally cache large ML model loading functions in memory using `functools.lru_cache(maxsize=1)` to prevent disk thrashing and redundant memory allocation.
