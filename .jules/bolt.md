
## 2024-05-24 - Cache ML Models in Memory
**Learning:** Large ML models (OpenAI Whisper and RVM) were being reloaded from disk on every invocation, causing severe performance bottlenecks.
**Action:** Always wrap heavy model loading functions with `@functools.lru_cache(maxsize=1)` to keep models in memory without risking OOM errors from multiple cached versions.
