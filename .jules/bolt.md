## 2026-09-13 - Cache Large ML Models in Memory
**Learning:** Repeatedly loading large ML models (OpenAI Whisper and RVM) from disk during sequential processing operations creates a severe performance bottleneck and risks Out-Of-Memory (OOM) issues.
**Action:** Used functools.lru_cache(maxsize=1) to globally cache model loading functions, drastically improving processing times and stability for subsequent jobs.
