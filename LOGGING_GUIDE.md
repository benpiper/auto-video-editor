# Enhanced Logging for Filler Word and Silence Detection

## Overview

The video processor now includes detailed logging that shows:
1. Each silence interval detected with timestamps
2. Each filler word detected with timestamps
3. Merged removal segments (after combining silence + filler words)
4. Summary statistics showing time saved

## Example Log Output

```
2025-11-25 10:18:00 - INFO - Extracting audio from input.mp4 to temp_audio.wav
2025-11-25 10:18:05 - INFO - Detecting silence...
2025-11-25 10:18:07 - INFO - Found 15 silence intervals:
2025-11-25 10:18:07 - INFO -   Silence 1: 12.50s - 15.30s (duration: 2.80s)
2025-11-25 10:18:07 - INFO -   Silence 2: 45.20s - 48.10s (duration: 2.90s)
2025-11-25 10:18:07 - INFO -   Silence 3: 89.40s - 92.00s (duration: 2.60s)
...

2025-11-25 10:18:07 - INFO - Loading Whisper model (base)...
2025-11-25 10:18:10 - INFO - Transcribing audio for filler word detection...
2025-11-25 10:18:45 - INFO -   Filler word detected: 'um' at 5.23s - 5.45s (duration: 0.22s)
2025-11-25 10:18:45 - INFO -   Filler word detected: 'uh' at 18.67s - 18.82s (duration: 0.15s)
2025-11-25 10:18:45 - INFO -   Filler word detected: 'um' at 34.12s - 34.38s (duration: 0.26s)
2025-11-25 10:18:45 - INFO -   Filler word detected: 'ah' at 56.90s - 57.05s (duration: 0.15s)
...
2025-11-25 10:18:45 - INFO - Found 23 filler words total.

2025-11-25 10:18:45 - INFO - Total intervals to remove: 15 silence + 23 filler words = 38
2025-11-25 10:18:45 - INFO - After merging overlapping intervals: 32 removal segments

2025-11-25 10:18:45 - INFO - Timestamp ranges to be removed:
2025-11-25 10:18:45 - INFO -   Segment 1: 5.23s - 5.45s (duration: 0.22s)
2025-11-25 10:18:45 - INFO -   Segment 2: 12.50s - 15.30s (duration: 2.80s)
2025-11-25 10:18:45 - INFO -   Segment 3: 18.67s - 18.82s (duration: 0.15s)
...
2025-11-25 10:18:45 - INFO - Total duration to be removed: 67.45s

2025-11-25 10:18:45 - INFO - Original video duration: 300.00s
2025-11-25 10:18:45 - INFO - Final video duration: 232.55s (removed 67.45s)

2025-11-25 10:18:45 - INFO - Cutting video. Keeping 32 segments.
2025-11-25 10:18:45 - INFO - Writing output to output.mp4
...
```

## Log Details

### Silence Detection
- Shows each silence interval found
- Includes start time, end time, and duration
- Helps identify if silence threshold is too aggressive or too lenient

### Filler Word Detection
- Shows the exact filler word detected ('um', 'uh', 'ah', etc.)
- Includes precise timestamps for each occurrence
- Helps verify that filler words are being correctly identified

### Merged Removal Segments
- Shows the final list of timestamp ranges to be removed
- Accounts for overlapping intervals (e.g., filler word during silence)
- Provides total duration that will be removed

### Summary Statistics
- Original video duration
- Final video duration after cuts
- Total time removed

## Use Cases

### 1. Verify Detection Accuracy
Check if the tool is correctly identifying silence and filler words:
```bash
uv run python main.py input.mp4 output.mp4 --min-silence 500 --silence-thresh -35 2>&1 | grep "detected"
```

### 2. See What's Being Removed
Review all removal segments:
```bash
uv run python main.py input.mp4 output.mp4 2>&1 | grep "Segment"
```

### 3. Get Summary Statistics
See how much time is being saved:
```bash
uv run python main.py input.mp4 output.mp4 2>&1 | grep "duration"
```

### 4. Save Full Log to File
Keep a record of all edits made:
```bash
uv run python main.py input.mp4 output.mp4 2>&1 | tee processing.log
```

## Adjusting Detection Parameters

Based on the log output, you can fine-tune the detection:

### Too Many Silence Intervals?
- Increase `--min-silence` (e.g., from 500 to 1000)
- Decrease `--silence-thresh` (e.g., from -35 to -40)

### Missing Some Silence?
- Decrease `--min-silence` (e.g., from 2000 to 1000)
- Increase `--silence-thresh` (e.g., from -40 to -35)

### Filler Words Not Detected?
- Check the log to see if Whisper is transcribing them
- Some filler words might be too subtle for Whisper to catch
- Consider using a larger Whisper model (though this is hardcoded to 'base')

## Example: Analyzing Your Video

For your network history lesson video:
```bash
uv run python main.py network-history-lesson.mp4 output.mp4 \
    --min-silence 500 --silence-thresh -35 --bitrate 5000k \
    2>&1 | tee network-lesson-processing.log
```

Then review the log:
```bash
# See all filler words detected
grep "Filler word detected" network-lesson-processing.log

# See all silence intervals
grep "Silence [0-9]" network-lesson-processing.log

# See summary
grep "duration:" network-lesson-processing.log | tail -3
```

## Benefits

1. **Transparency**: See exactly what's being removed and why
2. **Debugging**: Identify if parameters need adjustment
3. **Verification**: Confirm filler words are being caught
4. **Documentation**: Keep a record of edits for each video
5. **Quality Control**: Review removal decisions before final output
