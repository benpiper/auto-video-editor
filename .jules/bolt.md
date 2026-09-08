## 2026-09-08 - Cache large ML models globally in memory
**Learning:** Large ML models (e.g., OpenAI Whisper and RVM) cause severe performance bottlenecks from repeated disk loading and initialization overhead if not properly cached during processing, which can also lead to Out-Of-Memory (OOM) issues.
**Action:** Always wrap large ML model loading functions with `functools.lru_cache(maxsize=1)` to ensure models are globally cached in memory.
