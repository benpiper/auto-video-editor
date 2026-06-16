## 2024-10-24 - ML Model Loading Bottlenecks
**Learning:** Large ML models (Whisper and RVM) were being reloaded from disk on every invocation, causing significant IO bottlenecks. Naively caching globally risks Out-Of-Memory (OOM) errors.
**Action:** Use `functools.lru_cache(maxsize=1)` on model loader functions to maintain exactly one active model in memory across runs.
