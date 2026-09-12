## 2026-09-12 - Cache ML Models to Prevent Reloading
**Learning:** Repeatedly instantiating large ML models (like Whisper and RVM) without global caching leads to severe performance bottlenecks and OOM errors due to repeated disk loading during processing on each request.
**Action:** Use `functools.lru_cache(maxsize=1)` on model loading functions to maintain single global instances.
