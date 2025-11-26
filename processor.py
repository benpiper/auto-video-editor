import os
import logging
from typing import List, Tuple
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
from pydub import AudioSegment, silence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def extract_audio(video_path: str, audio_path: str):
    """Extracts audio from video file."""
    logging.info(f"Extracting audio from {video_path} to {audio_path}")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()

def detect_silence(audio_path: str, min_silence_len: int = 2000, silence_thresh: int = -40) -> List[Tuple[float, float]]:
    """
    Detects silence in audio file.
    Args:
        audio_path: Path to audio file.
        min_silence_len: Minimum length of silence in milliseconds.
        silence_thresh: Silence threshold in dBFS.
    Returns:
        List of (start, end) tuples in seconds.
    """
    logging.info("Detecting silence...")
    audio = AudioSegment.from_file(audio_path)
    # pydub returns intervals in milliseconds
    silence_intervals_ms = silence.detect_silence(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    # Convert to seconds
    silence_intervals_sec = [(start / 1000, end / 1000) for start, end in silence_intervals_ms]
    
    if silence_intervals_sec:
        logging.info(f"Found {len(silence_intervals_sec)} silence intervals:")
        for i, (start, end) in enumerate(silence_intervals_sec, 1):
            duration = end - start
            logging.info(f"  Silence {i}: {format_timestamp(start)} - {format_timestamp(end)} (duration: {duration:.2f}s)")
    else:
        logging.info("No silence intervals found.")
    
    return silence_intervals_sec

def detect_filler_words(audio_path: str, model_size: str = "base", use_crisper: bool = False) -> List[Tuple[float, float]]:
    """
    Detects filler words using Whisper or CrisperWhisper.
    Returns:
        List of (start, end) tuples in seconds.
    """
    if use_crisper:
        return detect_filler_words_crisper(audio_path)
    else:
        return detect_filler_words_whisper(audio_path, model_size)

def detect_filler_words_whisper(audio_path: str, model_size: str = "base") -> List[Tuple[float, float]]:
    """
    Detects filler words using standard Whisper.
    Returns:
        List of (start, end) tuples in seconds.
    """
    logging.info(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    logging.info("Transcribing audio for filler word detection...")
    # We use a prompt to encourage transcribing filler words if possible, though Whisper is trained to remove them.
    # Sometimes standard transcription removes them. We can try to rely on word-level timestamps.
    # temperature=0.0 for most literal/deterministic transcription
    # language='en' to force English detection
    result = model.transcribe(
        audio_path, 
        word_timestamps=True, 
        initial_prompt="um, uh, umm, uhh, er",
        temperature=0.0,
        language='en'
    )
    
    filler_words = ["um", "uh", "umm", "uhh", "er", "just, you know", "like, you know"]
    filler_intervals = []
    
    # Count total words for progress tracking
    total_words = sum(len(segment.get("words", [])) for segment in result["segments"])
    logging.info(f"Processing {total_words} transcribed words...")

    word_count = 0
    for segment in result["segments"]:
        for word in segment.get("words", []):
            word_count += 1
            word_text = word["word"].strip()
            word_start = word["start"]
            
            # Log every word with progress
            #logging.info(f"  Word {word_count}/{total_words}: [{word_start:.1f}s] \"{word_text}\"")
            
            # Check for filler words
            text = word_text.lower().replace(",", "").replace(".", "")
            if text in filler_words:
                start_time = word["start"]
                end_time = word["end"]
                duration = end_time - start_time
                filler_intervals.append((start_time, end_time))
                logging.info(f"    ✓ Filler word detected: '{text}' at {format_timestamp(start_time)} - {format_timestamp(end_time)} (duration: {duration:.2f}s)")
    
    logging.info(f"Found {len(filler_intervals)} filler words total.")
    
    # Save transcript to file
    transcript_text = result.get("text", "")
    if transcript_text:
        transcript_path = audio_path.replace(".wav", "_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        logging.info(f"Transcript saved to: {transcript_path}")
    
    return filler_intervals

def detect_filler_words_crisper(audio_path: str) -> List[Tuple[float, float]]:
    """
    Detects filler words using OpenAI Whisper Tiny model via HuggingFace.
    Falls back to standard Whisper if this fails.
    Returns:
        List of (start, end) tuples in seconds.
    """
    try:
        import torch
        from transformers import pipeline
        import soundfile as sf
        import numpy as np
        
        logging.info("Loading Whisper Tiny model...")
        
        # Use GPU if available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-tiny"
        
        # Create pipeline - it will load the model automatically
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps='word',
            dtype=torch_dtype,
            device=device,
            language='en',
        )
        
        logging.info(f"Transcribing audio with Whisper Tiny (device: {device})...")
        
        # Load audio and ensure it's in the right format
        audio_data, sample_rate = sf.read(audio_path)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Ensure float32 format
        audio_data = audio_data.astype(np.float32)
        
        # Transcribe - pipeline will handle resampling to 16kHz automatically
        result = pipe(
            {"array": audio_data, "sampling_rate": sample_rate},
            generate_kwargs={"language": "en", "task": "transcribe"}
        )
        
        # Calculate actual audio duration
        audio_duration = len(audio_data) / sample_rate
        logging.info(f"Audio duration: {format_timestamp(audio_duration)}")
        
        # Extract filler words
        filler_words = ["um", "uh", "umm", "uhh", "er"]
        filler_intervals = []
        
        if "chunks" in result:
            logging.info(f"Processing {len(result['chunks'])} transcribed chunks...")
            for i, chunk in enumerate(result["chunks"], 1):
                # Log transcription progress
                chunk_text = chunk["text"].strip()
                chunk_start = chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0
                #logging.info(f"  Chunk {i}: [{chunk_start:.1f}s] \"{chunk_text}\"")
                
                # Check for filler words
                text = chunk_text.lower().replace(",", "").replace(".", "")
                if text in filler_words:
                    start_time = chunk["timestamp"][0]
                    end_time = chunk["timestamp"][1] if chunk["timestamp"][1] is not None else start_time + 0.5
                    
                    # Validate timestamps - skip if beyond audio duration
                    if start_time >= audio_duration:
                        logging.warning(f"    ⚠ Skipping invalid filler word '{text}' at {format_timestamp(start_time)} (beyond audio duration {format_timestamp(audio_duration)})")
                        continue
                    
                    # Clamp end_time to audio duration
                    if end_time > audio_duration:
                        logging.warning(f"    ⚠ Clamping filler word '{text}' end time from {format_timestamp(end_time)} to {format_timestamp(audio_duration)}")
                        end_time = audio_duration
                    
                    duration = end_time - start_time
                    filler_intervals.append((start_time, end_time))
                    logging.info(f"    ✓ Filler word detected: '{text}' at {format_timestamp(start_time)} - {format_timestamp(end_time)} (duration: {duration:.2f}s)")
        
        # Clean up GPU memory
        if device == "cuda:0":
            del pipe
            torch.cuda.empty_cache()
        
        logging.info(f"Found {len(filler_intervals)} filler words total (Whisper Tiny).")
        
        # Save transcript to file
        transcript_text = result.get("text", "")
        if transcript_text:
            transcript_path = audio_path.replace(".wav", "_transcript_tiny.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            logging.info(f"Transcript saved to: {transcript_path}")
        
        return filler_intervals
        
    except ImportError as e:
        logging.error(f"Whisper Tiny requires transformers and datasets libraries: {e}")
        logging.error("Install with: pip install transformers datasets soundfile torchaudio librosa")
        logging.warning("Falling back to standard Whisper...")
        return detect_filler_words_whisper(audio_path, "base")
    except Exception as e:
        logging.error(f"Error using Whisper Tiny: {e}")
        logging.warning("Falling back to standard Whisper...")
        return detect_filler_words_whisper(audio_path, "base")




def merge_intervals(intervals: List[Tuple[float, float]], min_gap: float = 0.1) -> List[Tuple[float, float]]:
    """Merges overlapping or close intervals."""
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        
        if current_start <= last_end + min_gap:
            # Overlap or close enough to merge
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
            
    return merged

def invert_intervals(intervals: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
    """
    Inverts intervals to get the parts we want to KEEP.
    intervals: List of (start, end) to REMOVE.
    """
    keep_intervals = []
    current_time = 0.0
    
    for start, end in intervals:
        if start > current_time:
            keep_intervals.append((current_time, start))
        current_time = max(current_time, end)
        
    if current_time < total_duration:
        keep_intervals.append((current_time, total_duration))
    return keep_intervals

def get_encoding_params(use_gpu: bool, use_crf: bool, bitrate: str, crf: int, preset: str):
    """
    Get encoding parameters for CPU or GPU encoding.
    
    Args:
        use_gpu: Whether to use GPU (NVENC) encoding
        use_crf: Whether to use CRF mode (only for CPU)
        bitrate: Target bitrate
        crf: CRF value for quality
        preset: Encoding preset
    
    Returns:
        Tuple of (codec, preset_value, ffmpeg_params)
    """
    if use_gpu:
        # GPU encoding with NVENC
        # Check if NVENC is available (requires FFmpeg with NVENC support)
        logging.info("GPU encoding requested - checking NVENC availability...")
        
        # Try to detect if NVENC is available
        import subprocess
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if 'h264_nvenc' not in result.stdout:
                logging.warning("NVENC encoder not found in FFmpeg. Falling back to CPU encoding.")
                logging.warning("To enable GPU encoding, install FFmpeg with NVENC support.")
                use_gpu = False
        except Exception as e:
            logging.warning(f"Could not check for NVENC: {e}. Falling back to CPU encoding.")
            use_gpu = False
    
    if use_gpu:
        logging.info("Using NVIDIA GPU (NVENC) for video encoding")
        codec = "h264_nvenc"
        
        # NVENC presets: slow, medium, fast, hp, hq, bd, ll, llhq, llhp, lossless
        # Map x264 presets to NVENC presets
        nvenc_preset_map = {
            "veryslow": "hq",
            "slower": "hq",
            "slow": "hq",
            "medium": "medium",
            "fast": "fast",
            "faster": "fast",
            "veryfast": "fast",
            "superfast": "hp",
            "ultrafast": "hp"
        }
        nvenc_preset = nvenc_preset_map.get(preset, "medium")
        
        # For GPU encoding, keep it simple
        # MoviePy doesn't handle NVENC-specific parameters well
        # Just use bitrate - NVENC will handle it efficiently
        ffmpeg_params = []
        
        return codec, nvenc_preset, ffmpeg_params
    else:
        # CPU encoding with x264
        logging.info("Using CPU (x264) for video encoding")
        codec = "libx264"
        
        if use_crf:
            # CRF mode: quality-based, variable bitrate
            ffmpeg_params = [
                '-crf', str(crf),
                '-maxrate', bitrate,
                '-bufsize', f'{int(bitrate.rstrip("k")) * 2}k'
            ]
        else:
            # Constant bitrate mode: predictable file size
            ffmpeg_params = ['-b:v', bitrate]
        
        return codec, preset, ffmpeg_params



def get_video_rotation(video_path: str) -> int:
    """
    Get the rotation metadata from a video file using ffprobe.
    Returns rotation in degrees (0, 90, 180, 270).
    """
    import subprocess
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream_tags=rotate', '-of', 'default=nw=1:nk=1',
             video_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        rotation = result.stdout.strip()
        return int(rotation) if rotation else 0
    except:
        return 0

def transpose_video_if_needed(input_path: str, rotation: int) -> str:
    """
    If video has rotation metadata, create a transposed version using FFmpeg.
    Returns path to the (possibly transposed) video file.
    """
    if rotation == 0:
        return input_path
    
    import subprocess
    import tempfile
    
    # Create temporary file for transposed video
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4', prefix='transposed_')
    os.close(temp_fd)
    
    # FFmpeg transpose filter values:
    # 0 = 90° counterclockwise and vertical flip
    # 1 = 90° clockwise  
    # 2 = 90° counterclockwise
    # 3 = 90° clockwise and vertical flip
    
    transpose_map = {
        90: '2',    # 90° counterclockwise
        180: '2,transpose=2',  # 180° = two 90° rotations
        270: '1'    # 90° clockwise
    }
    
    transpose_filter = transpose_map.get(rotation)
    if not transpose_filter:
        return input_path
    
    logging.info(f"Transposing video {rotation}° using FFmpeg...")
    
    try:
        subprocess.run(
            ['ffmpeg', '-i', input_path, '-vf', f'transpose={transpose_filter}',
             '-c:a', 'copy', '-y', temp_path],
            capture_output=True,
            check=True,
            timeout=300
        )
        logging.info(f"Video transposed successfully to {temp_path}")
        return temp_path
    except Exception as e:
        logging.error(f"Failed to transpose video: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return input_path

def process_video(input_path: str, output_path: str, min_silence_len: int = 2000, silence_thresh: int = -63, crossfade_duration: float = 0.2, bitrate: str = "5000k", crf: int = 18, preset: str = "medium", use_crf: bool = False, use_gpu_encoding: bool = False, use_crisper_whisper: bool = False, no_crossfade: bool = False):
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return
    
    # Detect video rotation
    rotation = get_video_rotation(input_path)
    if rotation != 0:
        logging.info(f"Detected video rotation: {rotation} degrees")
    
    # Transpose video if needed to fix dimensions
    working_video_path = transpose_video_if_needed(input_path, rotation)
    transposed = (working_video_path != input_path)

    temp_audio_path = "temp_audio.wav"
    
    try:
        # 1. Extract Audio (from transposed video if applicable)
        extract_audio(working_video_path, temp_audio_path)
        
        # 2. Detect Silence
        silence_intervals = detect_silence(temp_audio_path, min_silence_len, silence_thresh)
        
        # 3. Detect Filler Words
        filler_intervals = detect_filler_words(temp_audio_path, use_crisper=use_crisper_whisper)
        
        # 4. Combine and Merge Intervals to Remove
        all_remove_intervals = silence_intervals + filler_intervals
        logging.info(f"Total intervals to remove: {len(silence_intervals)} silence + {len(filler_intervals)} filler words = {len(all_remove_intervals)}")
        
        merged_remove_intervals = merge_intervals(all_remove_intervals)
        logging.info(f"After merging overlapping intervals: {len(merged_remove_intervals)} removal segments")
        
        # Log each merged removal interval
        if merged_remove_intervals:
            logging.info("Timestamp ranges to be removed:")
            total_removed_duration = 0
            for i, (start, end) in enumerate(merged_remove_intervals, 1):
                duration = end - start
                total_removed_duration += duration
                logging.info(f"  Segment {i}: {format_timestamp(start)} - {format_timestamp(end)} (duration: {duration:.2f}s)")
            logging.info(f"Total duration to be removed: {format_timestamp(total_removed_duration)}")
        
        # 5. Get Keep Intervals
        video = VideoFileClip(working_video_path)  # Use transposed video if applicable
        total_duration = video.duration
        
        # Store original dimensions to preserve them
        original_size = video.size
        logging.info(f"Original video dimensions: {original_size[0]}x{original_size[1]} (width x height)")
        
        keep_intervals = invert_intervals(merged_remove_intervals, total_duration)
        
        logging.info(f"Original video duration: {format_timestamp(total_duration)}")
        if merged_remove_intervals:
            final_duration = sum(end - start for start, end in keep_intervals)
            logging.info(f"Final video duration: {format_timestamp(final_duration)} (removed {format_timestamp(total_duration - final_duration)})")
        
        if len(keep_intervals) == 1 and keep_intervals[0] == (0.0, total_duration):
            logging.info("✅ No cuts needed - no silence or filler words detected!")
            logging.info("The video is already optimized. Exiting without re-encoding.")
            video.close()
            return

        logging.info(f"Cutting video. Keeping {len(keep_intervals)} segments.")
        
        # 6. Create Subclips and Concatenate with Crossfade
        clips = []
        for start, end in keep_intervals:
            # Add a small buffer if needed, but for now exact cuts
            clip = video.subclip(start, end)
            clips.append(clip)
            
        # Apply crossfade
        # MoviePy's concatenate_videoclips with padding/method='compose' can do crossfades but it's tricky.
        # A simpler way for audio crossfade is `clip.audio.fadeout`. 
        # For video crossfade, we need overlapping clips.
        
        # Let's try a simple concatenation first. If crossfade is strictly required, we need to overlap.
        # To do crossfade: clip2 starts fading in while clip1 fades out.
        # We can use `composite_videoclips` or just `concatenate_videoclips` with padding.
        # However, simple concatenation is much faster and less error prone.
        # The user requested "smooth fade transition".
        
        final_clips = []
        for i, clip in enumerate(clips):
            # Apply crossfade only if not disabled
            if not no_crossfade and i > 0 and crossfade_duration > 0:
                clip = clip.crossfadein(crossfade_duration)
            
            final_clips.append(clip)
        
        # Check if we have any clips to concatenate
        if not final_clips:
            logging.error("No video segments to keep - entire video was removed!")
            logging.error("Try adjusting silence detection parameters or check if video has any content.")
            raise ValueError("No video content remaining after removing silence and filler words")
            
        # Use method='compose' for crossfades, 'chain' for simple concatenation
        if no_crossfade:
            logging.info("Using simple concatenation (no crossfades) for faster processing")
            final_video = concatenate_videoclips(final_clips, method='chain')
        else:
            logging.info("Using crossfade transitions")
            # Use method='compose' to enable crossfades
            # Note: This may cause slight dimension changes on some videos
            final_video = concatenate_videoclips(final_clips, method='compose', padding=-crossfade_duration if crossfade_duration > 0 else 0)
        
        logging.info(f"Writing output to {output_path}")
        
        # Get encoding parameters (GPU or CPU)
        codec, preset_value, ffmpeg_params = get_encoding_params(
            use_gpu_encoding, use_crf, bitrate, crf, preset
        )
        
        final_video.write_videofile(
            output_path,
            codec=codec,
            audio_codec="aac",
            bitrate=bitrate if use_gpu_encoding else None,
            preset=preset_value,
            audio_bitrate="192k",
            ffmpeg_params=ffmpeg_params
        )
        
        video.close()
        final_video.close()
        
    except KeyboardInterrupt:
        logging.warning("Processing interrupted by user (Ctrl+C)")
        # Clean up incomplete output file
        if os.path.exists(output_path):
            logging.info(f"Removing incomplete output file: {output_path}")
            try:
                os.remove(output_path)
            except Exception as cleanup_error:
                logging.error(f"Failed to remove incomplete file: {cleanup_error}")
        raise  # Re-raise to exit gracefully
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        # Clean up incomplete output file
        if os.path.exists(output_path):
            logging.info(f"Removing incomplete output file: {output_path}")
            try:
                os.remove(output_path)
            except Exception as cleanup_error:
                logging.error(f"Failed to remove incomplete file: {cleanup_error}")
                
    finally:
        # Cleanup
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        # Cleanup transposed video if created
        if transposed and os.path.exists(working_video_path):
            logging.info(f"Cleaning up transposed video: {working_video_path}")
            os.remove(working_video_path)
        
        logging.info("Done.")
if __name__ == "__main__":
    # For testing
    pass
