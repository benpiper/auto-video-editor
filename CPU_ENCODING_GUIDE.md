# CPU Encoding and Multi-Core Performance

## Multi-Core Support

**Yes, x264 (CPU encoding) uses multiple cores by default!**

### How It Works

The `libx264` encoder automatically detects and uses all available CPU cores for encoding. The number of threads is determined by:

1. **Automatic detection**: x264 detects your CPU core count
2. **Default behavior**: Uses `cores + 1` threads (e.g., 8 cores = 9 threads)
3. **No configuration needed**: Works out of the box

### Your System

Based on typical configurations, if you have:
- **8 CPU cores**: x264 will use ~9 threads
- **16 CPU cores**: x264 will use ~17 threads
- **32 CPU cores**: x264 will use ~33 threads

## Performance Optimization

### Preset Impact on Speed

The `--preset` parameter controls encoding speed vs compression efficiency:

| Preset | Speed | CPU Usage | Quality | Use Case |
|--------|-------|-----------|---------|----------|
| `ultrafast` | Fastest | Low | Lower | Quick previews |
| `veryfast` | Very Fast | Medium | Good | Fast processing |
| `fast` | Fast | Medium-High | Good | Balanced |
| **`medium`** | **Balanced** | **High** | **Very Good** | **Default** |
| `slow` | Slow | Very High | Excellent | Final output |
| `slower` | Very Slow | Maximum | Excellent | Archival |
| `veryslow` | Slowest | Maximum | Best | Maximum quality |

### Recommended Settings

**For your 15-minute video:**

```bash
# Fast processing (~5-10 minutes encoding)
uv run python main.py input.mp4 output.mp4 \
    --preset veryfast \
    --bitrate 5000k

# Balanced (default, ~10-20 minutes encoding)
uv run python main.py input.mp4 output.mp4 \
    --preset medium \
    --bitrate 5000k

# Best quality (~20-40 minutes encoding)
uv run python main.py input.mp4 output.mp4 \
    --preset slow \
    --use-crf --crf 18
```

## CPU vs GPU Encoding

### Why GPU Encoding Failed

Your FFmpeg installation doesn't have NVENC support compiled in. This is common with system-installed FFmpeg on Ubuntu/Debian.

### CPU Encoding Advantages

**Pros:**
- ✅ Better quality at same bitrate
- ✅ More control over encoding parameters
- ✅ Works on any system
- ✅ Uses all CPU cores automatically

**Cons:**
- ❌ Slower than GPU (2-5x)
- ❌ Higher CPU usage

### Performance Comparison

For your 15-minute video (905 seconds → 820 seconds after trimming):

| Method | Time | Quality |
|--------|------|---------|
| CPU `veryfast` | ~5-8 min | Good |
| CPU `medium` | ~10-15 min | Very Good |
| CPU `slow` | ~20-30 min | Excellent |
| GPU (if available) | ~3-5 min | Very Good |

## Monitoring CPU Usage

### Check CPU Usage During Encoding

```bash
# In a separate terminal
htop
```

You should see:
- **All CPU cores at ~100%** during video encoding
- **One core at ~100%** during Whisper transcription (GPU accelerated)

### Expected Timeline

For your 15-minute video with `--preset medium`:

1. **Audio extraction**: ~10 seconds
2. **Silence detection**: ~5 seconds
3. **Whisper transcription**: ~25 seconds (GPU accelerated)
4. **Video cutting**: ~10 seconds
5. **Video encoding**: ~10-15 minutes (CPU, all cores)

**Total**: ~12-18 minutes

## Optimizing for Speed

### Option 1: Use Faster Preset

```bash
uv run python main.py input.mp4 output.mp4 \
    --min-silence 1500 \
    --silence-thresh -40 \
    --preset veryfast \
    --bitrate 5000k
```

**Time saved**: ~50% faster encoding (5-8 minutes instead of 10-15)
**Quality impact**: Minimal for screen recordings

### Option 2: Lower Bitrate

```bash
uv run python main.py input.mp4 output.mp4 \
    --min-silence 1500 \
    --silence-thresh -40 \
    --bitrate 3000k
```

**Time saved**: ~20% faster encoding
**Quality impact**: Still very good for screen recordings

### Option 3: Combine Both

```bash
uv run python main.py input.mp4 output.mp4 \
    --min-silence 1500 \
    --silence-thresh -40 \
    --preset veryfast \
    --bitrate 3000k
```

**Time saved**: ~60% faster (4-6 minutes total encoding)
**Quality impact**: Good for most use cases

## Current Status

✅ **GPU encoding now has automatic fallback**
- If you use `--use-gpu-encoding` and NVENC isn't available, it automatically falls back to CPU
- You'll see a warning message in the logs
- No errors, just continues with CPU encoding

✅ **CPU encoding uses all cores**
- No configuration needed
- Automatic thread detection
- Maximum CPU utilization

## Recommendations

### For Quick Processing
```bash
uv run python main.py ghost-jobs.mp4 ghost-jobs-trimmed.mp4 \
    --min-silence 1500 \
    --silence-thresh -40 \
    --crossfade 0.2 \
    --preset veryfast \
    --bitrate 5000k
```

### For Best Quality
```bash
uv run python main.py ghost-jobs.mp4 ghost-jobs-trimmed.mp4 \
    --min-silence 1500 \
    --silence-thresh -40 \
    --crossfade 0.2 \
    --preset slow \
    --use-crf --crf 18
```

### For Balanced (Recommended)
```bash
uv run python main.py ghost-jobs.mp4 ghost-jobs-trimmed.mp4 \
    --min-silence 1500 \
    --silence-thresh -40 \
    --crossfade 0.2 \
    --bitrate 5000k
```

## Installing FFmpeg with NVENC (Optional)

If you want GPU encoding in the future:

```bash
# Option 1: Use conda (recommended)
conda install -c conda-forge ffmpeg

# Option 2: Build from source with NVENC support
# (Complex, not recommended unless necessary)
```

After installing FFmpeg with NVENC support, `--use-gpu-encoding` will work automatically!
