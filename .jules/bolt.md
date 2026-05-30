## 2025-05-30 - Caching Heavy PyTorch Models
**Learning:** Loading large Machine Learning models like Whisper and RVM from disk using `whisper.load_model` or `torch.hub.load` introduces significant performance overhead and latency (often multiple seconds) during repeated processing tasks.
**Action:** Use Python's `functools.lru_cache` to wrap model loading functions and keep these large objects cached in memory, ensuring subsequent calls for the same model type load in micro-seconds rather than seconds. Ensure `maxsize` is kept small to avoid excessive VRAM/RAM consumption.
