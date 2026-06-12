## 2024-06-12 - Memory Caching for Large ML Models
**Learning:** Large ML models like Whisper and RVM cause severe performance bottlenecks and potential OOM issues if repeatedly loaded from disk on every function call during processing pipelines.
**Action:** Always use `functools.lru_cache(maxsize=1)` to globally cache large models in memory to prevent repeated disk loading while guarding against Out-Of-Memory (OOM) issues.
