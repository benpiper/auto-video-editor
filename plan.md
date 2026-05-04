1. **Goal**: Make video processing faster by reducing unnecessary overhead.
2. **Current Bottleneck**: `processor.py` uses `moviepy.editor.VideoFileClip` for two main tasks:
   a. Extracting audio (loading the whole video and saving the audio)
   b. Getting the video's total duration (`video.duration` and `video.size`) just to calculate `invert_intervals`.
3. **Problem with `VideoFileClip`**: `moviepy.editor.VideoFileClip` is slow for basic metadata extraction and audio extraction because it loads the video, parses a lot of frames, and initializes complex objects. Using raw FFmpeg commands (`ffprobe` and `ffmpeg`) is noticeably faster for these simple operations, as shown in previous benchmark tests.
4. **Action Items**:
   a. **Modify `extract_audio` in `processor.py`**:
      Replace the current `VideoFileClip` implementation with `subprocess.run` to call `ffprobe` (to check for audio) and `ffmpeg` (to extract audio).
   b. **Create `get_video_duration_and_size` in `processor.py`**:
      Create a new helper function using `ffprobe` to fetch the video duration, width, and height.
   c. **Modify `process_video` in `processor.py`**:
      Replace the use of `VideoFileClip(working_video_path)` around line 622 with a call to the new `get_video_duration_and_size` function. This avoids opening the video clip just for its metadata. Note: The `video.close()` down the line in `process_video` around `line 646` (and maybe others) will need to be cleaned up or checked.
