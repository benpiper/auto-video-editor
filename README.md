# Auto Video Editor

A Python tool that automatically detects and removes silence and filler words (like "um" and "uh") from video files, applying smooth crossfade transitions between cuts.

## Features

- **Silence Detection**: Removes silent segments based on audio amplitude.
- **Filler Word Removal**: Uses OpenAI's Whisper model to transcribe and identify filler words for removal.
- **Smooth Transitions**: Applies crossfades to audio and video to avoid jarring jump cuts.

## Installation

1. **Prerequisites**: Ensure you have Python 3.8+ installed. You also need `ffmpeg` installed on your system (though `moviepy` often handles this).

2. **Install Dependencies**:
   ```bash
   pip install "moviepy<2.0" openai-whisper pydub torch
   ```
   *Note: `moviepy<2.0` is required for compatibility.*

## Usage

Run the script from the command line:

```bash
python main.py <input_file> <output_file> [options]
```

### Example

```bash
python main.py my_video.mp4 edited_video.mp4 --min-silence 500 --silence-thresh -35
```

### Options

- `input_file`: Path to the source video.
- `output_file`: Path where the processed video will be saved.
- `--min-silence`: Minimum length of silence to be removed, in milliseconds (default: `2000`).
- `--silence-thresh`: The threshold (in dBFS) below which audio is considered silent (default: `-40`).
- `--crossfade`: Duration of the crossfade transition in seconds (default: `0.1`).

## How It Works

1. **Audio Extraction**: The audio is extracted from the video file.
2. **Analysis**:
   - `pydub` analyzes the audio to find silent intervals.
   - `whisper` transcribes the audio to find timestamps of filler words.
3. **Processing**: Overlapping removal intervals are merged.
4. **Editing**: The video is cut to keep only the desired segments, and these segments are concatenated with a crossfade effect using `moviepy`.
