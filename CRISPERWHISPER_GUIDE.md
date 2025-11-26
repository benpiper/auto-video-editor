# CrisperWhisper Integration

## Overview

CrisperWhisper is now available as an optional transcription model for better filler word detection. Unlike standard Whisper which removes filler words, CrisperWhisper provides **verbatim transcription** that captures "um", "uh", "like", and other disfluencies.

## Setup

### 1. Install Dependencies

```bash
uv add transformers datasets soundfile
```

### 2. HuggingFace Authentication

CrisperWhisper requires HuggingFace authentication:

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login (you'll need a HuggingFace account)
huggingface-cli login
```

### 3. Accept Model License

Visit https://huggingface.co/nyrahealth/CrisperWhisper and accept the model license.

## Usage

### Command Line

Add the `--use-crisper-whisper` flag:

```bash
uv run python main.py input.mp4 output.mp4 \
    --min-silence 1500 \
    --silence-thresh -63 \
    --use-crisper-whisper
```

### Comparison

**Standard Whisper:**
```bash
uv run python main.py video.mp4 output.mp4
# Often detects 0 filler words
```

**CrisperWhisper:**
```bash
uv run python main.py video.mp4 output.mp4 --use-crisper-whisper
# Detects actual filler words like "um", "uh", "like"
```

## Features

### Detected Filler Words

CrisperWhisper detects:
- "um", "uh", "umm", "uhh"
- "er", "ah"
- "like" (when used as filler)
- "you know" (when used as filler)

### GPU Acceleration

CrisperWhisper automatically uses GPU if available (same as standard Whisper).

## Fallback Behavior

If CrisperWhisper fails (missing dependencies, authentication issues, etc.), the system automatically falls back to standard Whisper with a warning message.

## Performance

**First Run:**
- Downloads model (~1-2 GB)
- Takes longer due to model loading

**Subsequent Runs:**
- Uses cached model
- Similar speed to standard Whisper

## Troubleshooting

### ImportError: transformers not found

```bash
uv add transformers datasets soundfile
```

### Authentication Error

```bash
huggingface-cli login
# Then accept the model license at the URL above
```

### Model Download Fails

Check your internet connection and HuggingFace authentication:
```bash
huggingface-cli whoami
```

## Example Output

**Standard Whisper:**
```
Found 0 filler words total.
```

**CrisperWhisper:**
```
Loading CrisperWhisper model...
Transcribing audio with CrisperWhisper...
  Filler word detected: 'um' at 5.23s - 5.45s (duration: 0.22s)
  Filler word detected: 'uh' at 12.67s - 12.89s (duration: 0.22s)
  Filler word detected: 'like' at 18.34s - 18.56s (duration: 0.22s)
Found 3 filler words total (CrisperWhisper).
```

## Recommendations

**Use CrisperWhisper when:**
- You want to remove filler words
- Your videos have natural speech with disfluencies
- You have HuggingFace authentication set up

**Use Standard Whisper when:**
- You only need silence removal
- You don't have HuggingFace access
- You want faster processing (no model download)
