## 2026-09-17 - Prevent repeated model loading in memory
**Learning:** Large ML models (e.g., OpenAI Whisper and RVM) load from disk into memory every time their processing functions are called. This causes severe performance bottlenecks from repeated disk loading and could lead to OOM issues.
**Action:** Globally cache the loaded models in memory using `functools.lru_cache(maxsize=1)` to ensure models are only loaded once per process.
