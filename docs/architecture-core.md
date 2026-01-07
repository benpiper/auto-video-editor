# Architecture: Core Video Processor

The Core Processor is a modular Python package responsible for the heavy lifting of video analysis and editing.

## 1. Executive Summary
The Core Processor uses a combination of signal processing (audio) and machine learning (transcription/segmentation) to identify "removable" segments in a video. It then orchestrates FFmpeg to perform non-linear edits without requiring manual timeline work.

## 2. Technology Stack
- **Languages**: Python 3.10+
- **Signal Processing**: Pydub, Librosa
- **Computer Vision/ML**: OpenCV, OpenAI Whisper, MediaPipe, Robust Video Matting (RVM)
- **Video Rendering**: MoviePy / FFmpeg

## 3. Architecture Pattern
**Pipeline Architecture**: Data flows through sequential stages of analysis and transformation.
1.  **Stage 1: Ingestion**: Extract audio from source.
2.  **Stage 2: Analysis (Parallelizable)**:
    *   `SilenceDetector`: Amplitude-based scanning.
    *   `TranscriptionEngine`: Whisper-based word timing.
    *   `ScreenScanner`: Visual freeze-frame detection.
3.  **Stage 3: Reconciliation**: Merge overlapping removal intervals into a clean timeline.
4.  **Stage 4: Rendering**: Execute cuts and apply optional filters (e.g., background removal) via FFmpeg.

## 4. Key Components
- `processor.py`: Main orchestrator.
- `background_remover.py`: Encapsulates RVM logic.
- `person_segmenter.py`: Encapsulates MediaPipe logic.
- `create_test_video.py`: Synthetic data generator.
