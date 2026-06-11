
## 2024-06-11 - Caching Large ML Models Safely
**Learning:** Large ML models (e.g., OpenAI Whisper and RVM) cause severe performance bottlenecks when repeatedly loaded from disk. Using global dictionaries for caching them can lead to Out-Of-Memory (OOM) issues.
**Action:** Prefer using `functools.lru_cache(maxsize=1)` on model loading functions instead of global dictionaries to ensure models are cached efficiently while guarding against potential Out-Of-Memory (OOM) issues.
