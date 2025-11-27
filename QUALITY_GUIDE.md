# Video Quality Settings Guide

## Problem
The original implementation using MoviePy was producing low-quality output videos with significantly reduced bitrate and was very slow.

## Solution
Switched to direct **FFmpeg encoding** for both speed and quality:
- **Speed**: Uses FFmpeg stream copying where possible and fast encoding presets
- **Quality**: Configurable bitrate, CRF, and presets
- **Precision**: Frame-accurate cuts using FFmpeg

### New Command-Line Arguments

```bash
--bitrate BITRATE    # Video bitrate (default: 5000k)
--crf CRF           # CRF quality 0-51, lower = better (default: 18)
--preset PRESET     # Encoding preset (default: medium)
```

### Quality Parameters Explained

#### 1. **Bitrate** (`--bitrate`)
- Controls the target bits per second
- Default: `5000k` (5 Mbps) - much higher than original to ensure quality
- Higher = better quality but larger file size
- Examples:
  - `2000k` - Lower quality, smaller files
  - `5000k` - High quality (default)
  - `10000k` - Very high quality

#### 2. **CRF (Constant Rate Factor)** (`--crf`)
- Range: 0-51 (lower = better quality)
- Default: `18` (visually lossless)
- This is the **most important quality setting**
- Recommended values:
  - `0` - Lossless (huge files)
  - `18` - Visually lossless (default, recommended)
  - `23` - Default x264 (good quality)
  - `28` - Lower quality
  - `51` - Worst quality

#### 3. **Preset** (`--preset`)
- Controls encoding speed vs compression efficiency
- Default: `medium`
- Options (from fastest to slowest):
  - `ultrafast` - Fastest, largest files
  - `superfast`
  - `veryfast`
  - `faster`
  - `fast`
  - `medium` - **Default, good balance**
  - `slow` - Better compression
  - `slower`
  - `veryslow` - Best compression, slowest

**Note**: Slower presets give better compression (smaller files at same quality) but take longer to encode.

## Usage Examples

### High Quality (Recommended)
```bash
uv run python main.py input.mp4 output.mp4 --bitrate 5000k --crf 18 --preset medium
```

### Maximum Quality (Slow)
```bash
uv run python main.py input.mp4 output.mp4 --bitrate 10000k --crf 15 --preset slow
```

### Fast Processing (Lower Quality)
```bash
uv run python main.py input.mp4 output.mp4 --bitrate 3000k --crf 23 --preset veryfast
```

### Match Original Quality
To match your original video's bitrate:
```bash
uv run python main.py input.mp4 output.mp4 --bitrate 760k --crf 23 --preset medium
```

### Visually Lossless (Best for Archival)
```bash
uv run python main.py input.mp4 output.mp4 --crf 18 --preset slow
```

## Technical Details

### How CRF and Bitrate Work Together
- **CRF** is the primary quality control (variable bitrate)
- **Bitrate** sets a target, but CRF can override it
- When both are specified, x264 tries to achieve the CRF quality while targeting the bitrate
- For best results, use CRF alone or CRF + bitrate

### File Size Impact
With the new defaults (bitrate=5000k, crf=18):
- Expect **larger file sizes** than the low-quality default
- But **much better visual quality**
- File size will be roughly proportional to bitrate

### Encoding Time
- GPU (RTX 3060) is used for **Whisper transcription only**
- Video encoding uses CPU (x264)
- Slower presets = longer encoding time but better compression

## Comparison

| Setting | Bitrate | CRF | Preset | Quality | Speed | File Size |
|---------|---------|-----|--------|---------|-------|-----------|
| Old Default | ~147k | 23 | medium | Poor | Fast | Small |
| New Default | 5000k | 18 | medium | Excellent | Medium | Large |
| Fast | 3000k | 23 | veryfast | Good | Fast | Medium |
| Best | 10000k | 15 | slow | Exceptional | Slow | Very Large |

## Recommendations

1. **For most users**: Use the defaults (no flags needed)
   - `--bitrate 5000k --crf 18 --preset medium`

2. **For quick tests**: Use fast preset
   - `--preset veryfast --crf 23`

3. **For archival/professional**: Use slow preset with low CRF
   - `--crf 15 --preset slow`

4. **To match original quality**: Check original bitrate with `exiftool` and match it
   - `--bitrate <original_bitrate> --crf 23`
