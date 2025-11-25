# GPU Encoding Guide (NVENC)

## Overview

The auto-video-editor now supports **NVIDIA GPU-accelerated encoding** using NVENC (NVIDIA Encoder). This can speed up video encoding by **2-5x** compared to CPU encoding.

## What Uses GPU?

### With `--use-gpu-encoding` Flag

| Component | Hardware | Speed |
|-----------|----------|-------|
| **Whisper Transcription** | GPU (CUDA) | 5-10x faster than CPU |
| **Video Encoding** | GPU (NVENC) | 2-5x faster than CPU |
| **Audio Processing** | CPU | N/A |

### Without `--use-gpu-encoding` Flag (Default)

| Component | Hardware | Speed |
|-----------|----------|-------|
| **Whisper Transcription** | GPU (CUDA) | 5-10x faster than CPU |
| **Video Encoding** | CPU (x264) | Baseline |
| **Audio Processing** | CPU | N/A |

## Usage

### Basic GPU Encoding

```bash
uv run python main.py input.mp4 output.mp4 --use-gpu-encoding
```

### GPU Encoding with Quality Settings

```bash
# High quality with GPU encoding
uv run python main.py input.mp4 output.mp4 \
    --use-gpu-encoding \
    --bitrate 5000k

# GPU encoding with CRF mode
uv run python main.py input.mp4 output.mp4 \
    --use-gpu-encoding \
    --use-crf \
    --crf 18

# Fast GPU encoding
uv run python main.py input.mp4 output.mp4 \
    --use-gpu-encoding \
    --preset fast \
    --bitrate 3000k
```

## NVENC Presets

The `--preset` parameter is automatically mapped to NVENC presets:

| x264 Preset | NVENC Preset | Description |
|-------------|--------------|-------------|
| `veryslow`, `slower`, `slow` | `hq` | High quality, slower |
| `medium` | `medium` | Balanced (default) |
| `fast`, `faster`, `veryfast` | `fast` | Fast encoding |
| `superfast`, `ultrafast` | `hp` | High performance, fastest |

## Quality Comparison

### CPU (x264) vs GPU (NVENC)

**At the same bitrate:**
- **x264**: Slightly better quality, slower encoding
- **NVENC**: Slightly lower quality, much faster encoding
- **Difference**: Usually imperceptible for most content

**Recommendations:**
- **For final output**: Use CPU encoding (`--preset slow`)
- **For quick previews**: Use GPU encoding (`--use-gpu-encoding --preset fast`)
- **For balanced workflow**: Use GPU encoding with `--bitrate 5000k` or higher

## Performance Benchmarks

Based on typical screen recording (1080p, 5fps):

### CPU Encoding (x264)
```bash
# Without GPU encoding
uv run python main.py video.mp4 output.mp4 --preset medium
```
- **Encoding speed**: ~1-2x realtime
- **42-minute video**: ~20-40 minutes to encode

### GPU Encoding (NVENC)
```bash
# With GPU encoding
uv run python main.py video.mp4 output.mp4 --use-gpu-encoding --preset medium
```
- **Encoding speed**: ~5-10x realtime
- **42-minute video**: ~4-8 minutes to encode

### Total Processing Time

For a 42-minute video:
- **Whisper transcription**: ~5-10 minutes (GPU)
- **CPU encoding**: ~20-40 minutes
- **GPU encoding**: ~4-8 minutes

**Total time saved with GPU encoding**: ~15-30 minutes per video!

## Requirements

### Hardware
- **NVIDIA GPU** with NVENC support
  - Most NVIDIA GPUs from GTX 600 series and newer
  - Your RTX 3060 is fully supported ✅

### Software
- **CUDA-compatible PyTorch** (already installed via UV)
- **FFmpeg with NVENC support** (usually included)

### Verify NVENC Support

Check if your system supports NVENC:
```bash
ffmpeg -hide_banner -encoders | grep nvenc
```

You should see output like:
```
 V..... h264_nvenc           NVIDIA NVENC H.264 encoder
 V..... hevc_nvenc           NVIDIA NVENC hevc encoder
```

## Troubleshooting

### Error: "Unknown encoder 'h264_nvenc'"

**Cause**: FFmpeg doesn't have NVENC support compiled in.

**Solution**: Install FFmpeg with NVENC support:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Or use conda
conda install -c conda-forge ffmpeg
```

### Error: "Cannot load nvcuda.dll" or CUDA errors

**Cause**: NVIDIA drivers not installed or outdated.

**Solution**: Update NVIDIA drivers:
- Download from [NVIDIA website](https://www.nvidia.com/download/index.aspx)
- Or use your system's driver manager

### GPU encoding is slower than expected

**Possible causes:**
1. **Bottleneck elsewhere**: Check if disk I/O is the bottleneck
2. **Low GPU utilization**: Use `nvtop` to monitor GPU usage
3. **Thermal throttling**: Check GPU temperature

**Solutions:**
- Use faster storage (SSD)
- Increase bitrate to utilize GPU more
- Ensure good GPU cooling

### Quality is noticeably worse with GPU encoding

**Solution**: Increase bitrate or use CRF mode:
```bash
# Higher bitrate
--use-gpu-encoding --bitrate 8000k

# Or use CRF mode
--use-gpu-encoding --use-crf --crf 15
```

## Best Practices

### For Screen Recordings
```bash
# Fast processing with good quality
uv run python main.py lecture.mp4 output.mp4 \
    --use-gpu-encoding \
    --bitrate 5000k \
    --min-silence 1500 \
    --silence-thresh -38
```

### For Camera Footage
```bash
# High quality with GPU acceleration
uv run python main.py video.mp4 output.mp4 \
    --use-gpu-encoding \
    --use-crf \
    --crf 18 \
    --preset medium
```

### For Quick Previews
```bash
# Maximum speed
uv run python main.py video.mp4 preview.mp4 \
    --use-gpu-encoding \
    --preset fast \
    --bitrate 2000k
```

### For Final Output (Best Quality)
```bash
# Use CPU encoding for maximum quality
uv run python main.py video.mp4 final.mp4 \
    --preset slow \
    --use-crf \
    --crf 15
```

## Monitoring GPU Usage

Use `nvtop` to monitor GPU usage during encoding:
```bash
# In a separate terminal
nvtop
```

You should see:
- **During Whisper**: High GPU compute usage (~80-100%)
- **During NVENC encoding**: High encoder usage, moderate compute usage
- **During CPU encoding**: Low/idle GPU usage

## Summary

| Scenario | Command | Speed | Quality |
|----------|---------|-------|---------|
| **Quick preview** | `--use-gpu-encoding --preset fast` | Fastest | Good |
| **Balanced** | `--use-gpu-encoding --bitrate 5000k` | Fast | Very Good |
| **High quality** | `--use-gpu-encoding --use-crf --crf 18` | Fast | Excellent |
| **Maximum quality** | `--preset slow --crf 15` (CPU) | Slow | Best |

**Recommendation**: Use `--use-gpu-encoding` for most workflows. The quality difference is minimal, and the time savings are significant!
