## 2026-09-16 - Prevent OOM and Speed Up Processing by Caching ML Models
**Learning:** Large ML models like OpenAI Whisper and RVM cause severe performance bottlenecks and risk Out-Of-Memory (OOM) issues if repeatedly loaded from disk for each inference.
**Action:** Always globally cache these large models in memory using `functools.lru_cache(maxsize=1)` to ensure they are loaded only once.
