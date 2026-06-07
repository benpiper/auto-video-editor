## 2025-01-23 - ML Model Caching Strategy
**Learning:** Repeatedly loading large ML models from disk causes significant performance overhead and latency during processing pipelines.
**Action:** Always wrap model loading functions with `@functools.lru_cache(maxsize=1)` to cache models globally while preventing Out-Of-Memory (OOM) issues by capping the cache size.
