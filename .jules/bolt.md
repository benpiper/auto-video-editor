
## 2025-05-23 - Repeated ML Model Loading Bottleneck
**Learning:** The video processing pipeline (`processor.py`) exhibited a performance bottleneck by loading the large OpenAI Whisper model into memory from disk on every single processing request instead of caching it, severely impacting overall throughput.
**Action:** When working with large ML models in backend processing tasks, always cache loaded models in memory (e.g., using a global dictionary) so they can be reused across multiple requests within the same process.
