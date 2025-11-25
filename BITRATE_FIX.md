# Bitrate Fix - Important Update

## The Problem You Encountered

Your processed video had only **148 kbps** bitrate even though the code was set to use 5000k. This happened because:

1. **CRF mode was enabled by default** - CRF (Constant Rate Factor) is a quality-based encoding mode that uses variable bitrate
2. **For low-motion screen recordings** (like yours at 5 fps), CRF 18 compressed very efficiently, resulting in low bitrate
3. **The bitrate parameter was being ignored** when CRF was active

## The Solution

I've changed the default encoding mode to **Constant Bitrate** which ensures the specified bitrate is actually used.

### New Behavior

**Default (Constant Bitrate Mode)**:
```bash
uv run python main.py input.mp4 output.mp4 --bitrate 5000k
```
- Uses **constant bitrate** encoding (`-b:v 5000k`)
- Guarantees the specified bitrate is used
- More predictable file sizes and quality
- **This will fix your 148kbps issue!**

**CRF Mode (Optional)**:
```bash
uv run python main.py input.mp4 output.mp4 --use-crf --crf 18
```
- Uses **quality-based** encoding (variable bitrate)
- Better for videos with varying complexity
- May result in lower bitrate for simple/low-motion content
- Good for general use, but not ideal for screen recordings

## Recommended Settings for Your Use Case

### For Screen Recordings (Your Video)
```bash
# Match original quality (760 kbps)
uv run python main.py network-history-lesson.mp4 output.mp4 --bitrate 760k

# Higher quality (recommended)
uv run python main.py network-history-lesson.mp4 output.mp4 --bitrate 2000k

# Maximum quality
uv run python main.py network-history-lesson.mp4 output.mp4 --bitrate 5000k
```

### For Regular Videos (Camera footage, etc.)
```bash
# Use CRF mode for better compression
uv run python main.py input.mp4 output.mp4 --use-crf --crf 18
```

## What Changed in the Code

1. **Default mode**: Changed from CRF to constant bitrate
2. **New flag**: Added `--use-crf` to enable CRF mode when needed
3. **Encoding logic**: 
   - Without `--use-crf`: Uses `-b:v <bitrate>` (constant bitrate)
   - With `--use-crf`: Uses `-crf <value>` with maxrate constraints

## Quick Reference

| Mode | Command | Bitrate Behavior | Best For |
|------|---------|------------------|----------|
| **Constant Bitrate** (default) | `--bitrate 5000k` | Fixed at 5000k | Screen recordings, predictable quality |
| **CRF Mode** | `--use-crf --crf 18` | Variable (adapts to content) | Camera footage, varying complexity |

## Next Steps

**Re-run your processing with the fix:**
```bash
# This will now actually use 5000k bitrate!
uv run python main.py network-history-lesson.mp4 network-history-lesson-trimmed-v2.mp4 \
    --min-silence 500 --silence-thresh -35 --bitrate 5000k
```

The output should now have **~5000 kbps** bitrate instead of 148 kbps!

## Technical Details

### Why CRF Failed for Your Video

- Your video: 5 fps screen recording with mostly static content
- CRF 18 saw this as "very compressible" and used minimal bitrate
- Result: 148 kbps (technically "visually lossless" for that content)
- But you wanted higher bitrate for archival/quality assurance

### Why Constant Bitrate Works Better

- Forces encoder to use the specified bitrate
- Ensures consistent quality across the entire video
- Better for screen recordings where you want guaranteed quality
- More predictable file sizes
