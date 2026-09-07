## 2026-09-07 - Globally Cache Large ML Models
**Learning:** Repeatedly loading large ML models from disk for each request causes severe performance bottlenecks and increases the risk of Out-Of-Memory (OOM) issues due to redundant allocations.
**Action:** Always globally cache large models in memory using `functools.lru_cache(maxsize=1)` to ensure only one instance resides in memory per model variant, significantly improving response times for subsequent processing.
