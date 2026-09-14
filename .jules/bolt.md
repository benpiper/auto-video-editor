
## 2026-09-14 - Caching Large ML Models Globally
**Learning:** Loading large ML models like Whisper and RVM from disk repeatedly for each request causes severe performance bottlenecks and can lead to Out-Of-Memory (OOM) issues due to memory leaks or multiple instances.
**Action:** Always wrap expensive ML model loading functions with `@functools.lru_cache(maxsize=1)` to ensure they are loaded once and shared globally across requests in the same process.
