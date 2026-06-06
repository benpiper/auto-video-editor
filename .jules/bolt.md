## 2024-05-20 - Global Caching for Large ML Models
**Learning:** Large ML models (like OpenAI Whisper and RVM) cause severe performance bottlenecks when repeatedly loaded from disk during processing. Using simple global dictionaries for caching can lead to Out-Of-Memory (OOM) issues.
**Action:** Use `functools.lru_cache` to globally cache these models in memory to prevent repeated disk loading and safely manage memory to avoid OOM issues.
