## 2024-10-24 - ML Model Caching Strategy
**Learning:** Repeatedly calling model load functions like `whisper.load_model` and `torch.hub.load` without caching forces the application to load gigabytes of weights from disk on every processing request, causing severe performance bottlenecks.
**Action:** Always wrap heavy ML model loading functions with `@functools.lru_cache(maxsize=1)` to keep the model in memory across requests, preventing redundant disk I/O while guarding against OOM errors.
