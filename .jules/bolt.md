## 2026-09-15 - Memory Caching for Large ML Models
**Learning:** Large ML models (like Whisper and RVM) cause severe performance bottlenecks and potential OOM issues when repeatedly loaded from disk during video processing jobs.
**Action:** Always globally cache large ML models in memory using `functools.lru_cache(maxsize=1)` to ensure they are instantiated only once.
