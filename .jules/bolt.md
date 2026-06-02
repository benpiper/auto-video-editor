## 2025-03-01 - Model Loading Overhead
**Learning:** Large machine learning models (like Whisper and RVM) have a significant initialization cost because they must be repeatedly loaded from disk if not cached. However, naïve global dictionary caching risks out-of-memory (OOM) errors over time.
**Action:** Always use `functools.lru_cache(maxsize=1)` or similar size-limited LRU caches for loading large models, preventing repeated slow disk reads while inherently capping memory usage to one active model instance.
