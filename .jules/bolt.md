## 2025-06-15 - Global Caching for Large ML Models
**Learning:** Large machine learning models like OpenAI Whisper and RVM cause severe performance bottlenecks if loaded from disk on every processing request.
**Action:** Use `functools.lru_cache(maxsize=1)` on model loading functions to keep them globally cached in memory. This avoids repeated disk loading and guarantees we don't hold multiple instances, preventing OOM issues.
