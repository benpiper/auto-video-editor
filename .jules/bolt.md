## 2026-09-09 - Cache ML models globally
**Learning:** Large ML models (like OpenAI Whisper and RVM) can cause severe performance bottlenecks and potential OOM issues due to repeated disk loading during processing if not cached in memory.
**Action:** Always use `functools.lru_cache(maxsize=1)` to cache large model loading functions to ensure they are loaded only once.
