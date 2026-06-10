## 2025-02-12 - ML Model Caching for OOM Prevention
**Learning:** Repeatedly loading large ML models (like Whisper and RVM) from disk for every request causes severe performance bottlenecks and memory fragmentation.
**Action:** Use `functools.lru_cache(maxsize=1)` to globally cache loaded models in memory, preventing repeated disk I/O while automatically managing potential Out-Of-Memory (OOM) issues when different models are requested.
