## 2024-05-31 - Cache ML models to prevent reloading
**Learning:** Loading large ML models (Whisper, RVM) from disk every time a video is processed introduces a severe performance bottleneck.
**Action:** Wrap model loading functions with `@functools.lru_cache(maxsize=1)` to load them only once per process, significantly reducing processing time on subsequent requests while preventing Out-Of-Memory (OOM) issues.
