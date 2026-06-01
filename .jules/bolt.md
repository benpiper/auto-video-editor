## 2024-05-24 - Model Caching for ML Workloads
**Learning:** Large ML models like OpenAI Whisper and RVM cause severe performance bottlenecks when repeatedly loaded from disk during multi-step video processing.
**Action:** Use `functools.lru_cache(maxsize=1)` to globally cache model instances in memory, preventing redundant I/O operations while guarding against Out-Of-Memory (OOM) issues.
