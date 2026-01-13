# Video Quality & Processing Guide 🎬

AutoCut AI offers two primary processing pipelines, optimized for different stages of the editing workflow.

## 🚀 Speed Mode (The "Rough Cut")
**Ideal for:** Quick verification, rough drafts, and high-frequency iteration.

- **Technology**: Uses `direct FFmpeg x264` or `NVENC` (hardware-accelerated) with hard cuts.
- **Method**: Implements frame-accurate segmentation. When possible, it uses stream-copying to avoid re-encoding entire segments.
- **Speed**: Usually renders at a ratio < 0.2 (e.g., 10-minute video in < 2 minutes).
- **Transitions**: Uses "Hard Cuts" (no overlaps).

## ✨ Quality Mode (The "Final Master")
**Ideal for:** Final delivery, professional presentations, and social media exports.

- **Technology**: Powered by `MoviePy` for high-level clip composition and smoothing.
- **Method**: 
  - **Crossfades**: Automatically applies a 0.2s crossfade between every kept segment.
  - **Audio Smoothing**: Applies subtle audio fade-ins and fade-outs to every clip to prevent digital "pops" or jarring ambient sound shifts.
  - **High-Fidelity Encoding**: Defaults to `libx264 slower` or `h264_nvenc p7` with CRF 18 (visually lossless).
- **Speed**: Slower due to the complexity of blending layers and audio tracks.
- **Transitions**: 0.2s Cross-dissolves.

---

## 🛠️ Advanced Quality Parameters

| Parameter   | Default            | Purpose                                                                |
| :---------- | :----------------- | :--------------------------------------------------------------------- |
| `bitrate`   | `12000k` (Quality) | Maximum ceiling for video bitrate.                                     |
| `crf`       | `18`               | Constant Rate Factor. Lower = Higher Quality. 18 is visually lossless. |
| `preset`    | `medium`           | Encoding effort. Slower = Better compression/quality.                  |
| `crossfade` | `0.2s`             | Overlap duration between clips.                                        |

### How to choose?

- **Use Speed Mode** if you still need to review your cuts and sensitivity settings. It’s nearly 5-10x faster.
- **Use Quality Mode** once you have finalized your removal list and want a professional, "seamless" video with smooth transitions.

## ⚡ Hardware Acceleration (NVENC)
AutoCut AI automatically detects and utilizes NVIDIA GPUs. 
- **In Speed Mode**: Uses `h264_nvenc` with the `p1` preset for maximum throughput.
- **In Quality Mode**: Uses `h264_nvenc` with the `p7` (Highest Quality) preset and `hq` tuning.

## 🖥️ Command Line Examples

**High-Fidelity Master (Quality Mode equivalent):**
```bash
uv run python main.py input.mp4 output.mp4 --bitrate 12000k --crf 18 --preset slower
```

**Fastest Possible Render (Speed Mode equivalent):**
```bash
uv run python main.py input.mp4 output.mp4 --preset ultrafast --no-crossfade
```
