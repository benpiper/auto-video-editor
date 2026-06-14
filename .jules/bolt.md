## 2024-05-24 - Cache Large ML Models to Prevent Disk Reloads
**Learning:** Large models like OpenAI Whisper and RVM become severe performance bottlenecks when repeatedly loaded from disk into memory for each processing task.
**Action:** Use `functools.lru_cache(maxsize=1)` to globally cache these models in memory, preventing redundant disk loads while capping the cache size to avoid Out-Of-Memory (OOM) issues.
