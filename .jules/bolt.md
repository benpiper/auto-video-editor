## 2026-05-31 - Model Caching Prevents Disk Bottlenecks
**Learning:** Large ML models like OpenAI Whisper and RVM in this app must be globally cached in memory to prevent severe performance bottlenecks from repeated disk loading during processing. Using `functools.lru_cache` is preferred over global dictionaries to guard against potential Out-Of-Memory (OOM) issues.
**Action:** Apply `functools.lru_cache` to functions that load heavy ML models to optimize speed and avoid repeated disk I/O bottlenecks.
