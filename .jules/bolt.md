## 2026-06-08 - Cache Large ML Models to Prevent Repeated Loading
**Learning:** Large ML models like OpenAI Whisper and RVM cause severe performance bottlenecks if loaded from disk on every function call during processing.
**Action:** Use `functools.lru_cache(maxsize=1)` to globally cache these models in memory, preventing repeated disk I/O while guarding against Out-Of-Memory (OOM) issues.
