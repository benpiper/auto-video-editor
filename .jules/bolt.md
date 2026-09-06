
## 2026-09-06 - Cache large ML models globally
**Learning:** Large ML models (e.g., OpenAI Whisper and RVM) must be globally cached in memory using `functools.lru_cache(maxsize=1)` to prevent severe performance bottlenecks from repeated disk loading during processing and guard against potential Out-Of-Memory (OOM) issues.
**Action:** Always wrap large model loading functions with `@functools.lru_cache(maxsize=1)`.
