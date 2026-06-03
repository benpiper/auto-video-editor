## 2024-05-23 - Global Caching for ML Models
**Learning:** Large ML models (like OpenAI Whisper and RVM) were being loaded from disk on every function call, causing severe performance bottlenecks from repeated disk I/O.
**Action:** Always prefer `functools.lru_cache(maxsize=1)` for expensive model loading functions to globally cache them in memory and ensure fast subsequent inferences, while guarding against potential Out-Of-Memory (OOM) issues better than simple dictionaries.
