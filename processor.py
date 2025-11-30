import os
import logging
from typing import List, Tuple, Optional, Callable
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
    """Extracts audio from video file. Returns True if audio exists, False otherwise."""
    logging.info(f"Extracting audio from {video_path} to {audio_path}")
    video = VideoFileClip(video_path)
    
    if video.audio is None:
        logging.warning("Video has no audio track - skipping audio-based detection")
        video.close()
        return False
    
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    return True

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

def detect_filler_words(audio_path: str, model_size: str = "base", filler_words_list: List[str] = None) -> List[Tuple[float, float]]:
    """
    Detects filler words using Whisper.
    Returns:
        List of (start, end) tuples in seconds.
    """
    return detect_filler_words_whisper(audio_path, model_size, filler_words_list)

def detect_filler_words_whisper(audio_path: str, model_size: str = "base", filler_words_list: List[str] = None) -> List[Tuple[float, float]]:
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
    
    if filler_words_list:
        filler_words = [w.lower().strip() for w in filler_words_list]
        logging.info(f"Using custom filler words: {filler_words}")
    else:
        filler_words = ["um", "uh", "umm", "uhh", "er", "just", "you know", "like, you know"]
        logging.info(f"Using default filler words: {filler_words}")
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

def detect_freeze_frames(video_path: str, min_duration: float = 5.0, noise_tolerance: float = 0.001) -> List[Tuple[float, float]]:
    """
    Detects freeze/still frames using FFmpeg's freezedetect filter.
    
    Args:
        video_path: Path to video file
        min_duration: Minimum freeze duration in seconds to detect
        noise_tolerance: Noise tolerance (0.001 = very sensitive, 0.01 = less sensitive)
    
    Returns:
        List of (start, end) tuples in seconds for frozen intervals
    """
    import subprocess
    import re
    
    logging.info(f"Detecting freeze frames (min duration: {min_duration}s, noise tolerance: {noise_tolerance})...")
    
    try:
        # Run FFmpeg with freezedetect filter
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'freezedetect=n={noise_tolerance}:d={min_duration}',
            '-f', 'null',
            '-'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        # Parse freeze detection from stderr
        freeze_intervals = []
        freeze_start = None
        
        for line in result.stderr.split('\n'):
            # Look for freeze_start and freeze_end lines
            if 'freezedetect' in line:
                if 'freeze_start' in line:
                    # Extract timestamp: lavfi.freezedetect.freeze_start: 12.5
                    match = re.search(r'freeze_start:\s*([\d.]+)', line)
                    if match:
                        freeze_start = float(match.group(1))
                elif 'freeze_end' in line and freeze_start is not None:
                    # Extract timestamp: lavfi.freezedetect.freeze_end: 18.2
                    match = re.search(r'freeze_end:\s*([\d.]+)', line)
                    if match:
                        freeze_end = float(match.group(1))
                        freeze_intervals.append((freeze_start, freeze_end))
                        freeze_start = None
        
        if freeze_intervals:
            logging.info(f"Found {len(freeze_intervals)} freeze frame intervals:")
            for i, (start, end) in enumerate(freeze_intervals, 1):
                duration = end - start
                logging.info(f"  Freeze {i}: {format_timestamp(start)} - {format_timestamp(end)} (duration: {duration:.2f}s)")
        else:
            logging.info("No freeze frames detected.")
        
        return freeze_intervals
        
    except subprocess.TimeoutExpired:
        logging.error("Freeze detection timed out")
        return []
    except Exception as e:
        logging.error(f"Error detecting freeze frames: {e}")
        return []




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

def extract_segments_ffmpeg(input_path: str, segments: List[Tuple[float, float]], temp_dir: str = ".", target_bitrate: str = "4M", progress_callback: Optional[Callable[[int, str], None]] = None) -> List[str]:
    """
    Extract video segments using FFmpeg with re-encoding for precise cuts.
    
    Args:
        input_path: Path to input video
        segments: List of (start, end) tuples in seconds
        temp_dir: Directory for temporary segment files
        target_bitrate: Target video bitrate (e.g., "4M" for 4 Mbps)
    
    Returns:
        List of paths to extracted segment files
    """
    import subprocess
    import tempfile
    
    segment_files = []
    
    for i, (start, end) in enumerate(segments):
        duration = end - start
        segment_file = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
        
        logging.info(f"Extracting segment {i+1}/{len(segments)}: {format_timestamp(start)} - {format_timestamp(end)}")
        if progress_callback:
            # Map segment extraction to 60-90% range
            progress = 60 + int(30 * ((i + 1) / len(segments)))
            progress_callback(progress, f"Extracting segment {i+1}/{len(segments)}")
        
        try:
            # Use FFmpeg to extract segment with re-encoding for precise cuts
            # Note: Re-encoding is necessary to cut at exact timestamps (not just keyframes)
            # This ensures no extra silence is added at cut points
            cmd = [
                'ffmpeg',
                '-accurate_seek',             # Enable accurate seeking
                '-i', input_path,             # Input file
                '-ss', str(start),            # Seek to start (accurate, frame-level)
                '-t', str(duration),          # Duration
                '-c:v', 'libx264',            # Re-encode video for precise cuts
                '-preset', 'medium',          # Balanced encoding preset for quality
                '-b:v', target_bitrate,       # Target bitrate (from source detection)
                '-maxrate', target_bitrate,   # Max bitrate
                '-bufsize', f'{int(target_bitrate[:-1])*2}M',  # Buffer size (2x bitrate)
                '-c:a', 'aac',                # Re-encode audio
                '-b:a', '192k',               # Audio bitrate
                '-avoid_negative_ts', '1',    # Handle timestamp issues
                '-y',                         # Overwrite output
                segment_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800
            )
            
            if result.returncode != 0:
                logging.error(f"FFmpeg segment extraction failed: {result.stderr}")
                raise RuntimeError(f"Failed to extract segment {i}")
            
            segment_files.append(segment_file)
            
        except Exception as e:
            logging.error(f"Error extracting segment {i}: {e}")
            # Clean up any created files
            for f in segment_files:
                if os.path.exists(f):
                    os.remove(f)
            raise
    
    logging.info(f"Successfully extracted {len(segment_files)} segments")
    return segment_files

def concatenate_segments_ffmpeg(segment_files: List[str], output_path: str, codec: str, preset: str, ffmpeg_params: list, bitrate: str = None) -> None:
    """
    Concatenate video segments using FFmpeg.
    
    Args:
        segment_files: List of segment file paths
        output_path: Output video path
        codec: Video codec to use
        preset: Encoding preset
        ffmpeg_params: Additional FFmpeg parameters
        bitrate: Video bitrate (optional)
    """
    import subprocess
    import tempfile
    
    # Create concat file list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='.') as f:
        concat_file = f.name
        for segment in segment_files:
            # FFmpeg concat requires forward slashes and escaped special chars
            escaped_path = segment.replace('\\', '/').replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    
    try:
        logging.info(f"Concatenating {len(segment_files)} segments with FFmpeg...")
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
        ]
        
        # Add encoding parameters
        if codec == 'copy':
            # Stream copy - very fast, no re-encoding
            cmd.extend(['-c', 'copy'])
        else:
            # Re-encode with specified codec
            cmd.extend(['-c:v', codec])
            cmd.extend(['-c:a', 'aac'])
            
            if bitrate:
                cmd.extend(['-b:v', bitrate])
            
            if preset:
                cmd.extend(['-preset', preset])
            
            # Add any additional parameters
            if ffmpeg_params:
                cmd.extend(ffmpeg_params)
        
        cmd.extend(['-y', output_path])
        
        logging.info(f"FFmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200
        )
        
        if result.returncode != 0:
            logging.error(f"FFmpeg concatenation failed: {result.stderr}")
            raise RuntimeError("Failed to concatenate segments")
        
        logging.info(f"Successfully created output: {output_path}")
        
    finally:
        # Clean up concat file
        if os.path.exists(concat_file):
            os.remove(concat_file)

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
            timeout=1800
        )
        logging.info(f"Video transposed successfully to {temp_path}")
        return temp_path
    except Exception as e:
        logging.error(f"Failed to transpose video: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return input_path

def process_video(input_path: str, output_path: str, min_silence_len: int = 2000, silence_thresh: int = -63, crossfade_duration: float = 0.2, bitrate: str = "5000k", crf: int = 18, preset: str = "medium", use_crf: bool = False, use_gpu_encoding: bool = False, no_crossfade: bool = False, filler_words: List[str] = None, freeze_duration: float = None, freeze_noise: float = 0.001, remove_background: bool = False, bg_color: str = "green", bg_image: Optional[str] = None, rvm_model: str = "mobilenetv3", rvm_downsample: Optional[float] = None, progress_callback: Optional[Callable[[int, str], None]] = None):
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
    
    if progress_callback:
        progress_callback(0, "Starting processing...")
    
    try:
        # 1. Extract Audio (from transposed video if applicable)
        if progress_callback:
            progress_callback(5, "Extracting audio...")
        has_audio = extract_audio(working_video_path, temp_audio_path)
        
        # 2. Detect Silence (only if audio exists)
        silence_intervals = []
        if has_audio:
            if progress_callback:
                progress_callback(10, "Detecting silence...")
            silence_intervals = detect_silence(temp_audio_path, min_silence_len, silence_thresh)
        else:
            logging.info("Skipping silence detection (no audio)")
        
        # 3. Detect Filler Words (only if audio exists)
        filler_intervals = []
        if has_audio:
            if progress_callback:
                progress_callback(20, "Detecting filler words (this may take a while)...")
            filler_intervals = detect_filler_words(temp_audio_path, filler_words_list=filler_words)
        else:
            logging.info("Skipping filler word detection (no audio)")
        
        # 4. Detect Freeze Frames (if enabled)
        freeze_intervals = []
        if freeze_duration is not None and freeze_duration > 0:
            if progress_callback:
                progress_callback(40, "Detecting freeze frames...")
            freeze_intervals = detect_freeze_frames(working_video_path, min_duration=freeze_duration, noise_tolerance=freeze_noise)
        
        # 5. Combine and Merge Intervals to Remove
        if progress_callback:
            progress_callback(50, "Calculating cuts...")
        all_remove_intervals = silence_intervals + filler_intervals + freeze_intervals
        logging.info(f"Total intervals to remove: {len(silence_intervals)} silence + {len(filler_intervals)} filler words + {len(freeze_intervals)} freeze frames = {len(all_remove_intervals)}")
        
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
            
            # Check if background removal is requested
            if remove_background:
                logging.info("Background removal is enabled - will process the video for background removal only...")
                # Copy input to output, then apply background removal
                import shutil
                shutil.copy2(working_video_path, output_path)
                logging.info(f"Copied original video to: {output_path}")
                video.close()
                # Continue to background removal below
            else:
                logging.info("The video is already optimized. Exiting without re-encoding.")
                video.close()
                return

        logging.info(f"Cutting video. Keeping {len(keep_intervals)} segments.")
        
        # Detect source video bitrate
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
                 working_video_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                source_bitrate_bps = int(result.stdout.strip())
                source_bitrate_mbps = source_bitrate_bps / 1_000_000
                target_bitrate = f"{int(source_bitrate_mbps)}M"
                logging.info(f"Detected source bitrate: {source_bitrate_mbps:.2f} Mbps, using {target_bitrate} for output")
            else:
                target_bitrate = "4M"  # Default fallback
                logging.warning(f"Could not detect source bitrate, using default {target_bitrate}")
        except Exception as e:
            target_bitrate = "4M"  # Default fallback
            logging.warning(f"Error detecting bitrate: {e}, using default {target_bitrate}")
        
        # 6. Extract segments and concatenate with FFmpeg (much faster than MoviePy)
        segment_files = []
        try:
            # Extract segments using FFmpeg with detected bitrate
            if progress_callback:
                progress_callback(60, "Extracting video segments...")
            segment_files = extract_segments_ffmpeg(working_video_path, keep_intervals, temp_dir=".", target_bitrate=target_bitrate, progress_callback=progress_callback)
            
            # Get encoding parameters
            codec, preset_value, ffmpeg_params = get_encoding_params(
                use_gpu_encoding, use_crf, bitrate, crf, preset
            )
            
            # Decide whether to use stream copy or re-encode
            if no_crossfade:
                # Use stream copy for maximum speed (no re-encoding)
                logging.info("Using FFmpeg stream copy (no re-encoding) for maximum speed")
                if progress_callback:
                    progress_callback(95, "Concatenating segments...")
                concatenate_segments_ffmpeg(
                    segment_files,
                    output_path,
                    codec='copy',  # Stream copy
                    preset=None,
                    ffmpeg_params=[],
                    bitrate=None
                )
            else:
                # Re-encode with crossfades (slower but still faster than MoviePy)
                logging.warning("Crossfades with FFmpeg require re-encoding (slower)")
                logging.info("Consider using --no-crossfade for much faster processing")
                if progress_callback:
                    progress_callback(95, "Concatenating segments (this may take a while)...")
                concatenate_segments_ffmpeg(
                    segment_files,
                    output_path,
                    codec=codec,
                    preset=preset_value,
                    ffmpeg_params=ffmpeg_params,
                    bitrate=bitrate if use_gpu_encoding else None
                )
            
            logging.info(f"✅ Video processing complete: {output_path}")
            
        finally:
            # Clean up segment files
            for segment_file in segment_files:
                if os.path.exists(segment_file):
                    try:
                        os.remove(segment_file)
                        logging.debug(f"Removed temporary segment: {segment_file}")
                    except Exception as e:
                        logging.warning(f"Failed to remove temporary segment {segment_file}: {e}")
        
        # 7. Apply background removal if requested (post-processing)
        # This runs after video processing completes (whether cuts were made or not)
        if remove_background and os.path.exists(output_path):
            logging.info("Applying background removal...")
            if progress_callback:
                progress_callback(98, "Removing background (this may take a while)...")
            
            try:
                from background_remover import apply_background_removal
                
                # Create temporary file for background-removed video
                import tempfile
                temp_fd, temp_bg_removed = tempfile.mkstemp(suffix='.mp4', prefix='bg_removed_')
                os.close(temp_fd)
                
                # Apply background removal to the processed video
                apply_background_removal(
                    output_path,
                    temp_bg_removed,
                    model_name=rvm_model,
                    background_color=bg_color,
                    background_image=bg_image,
                    downsample_ratio=rvm_downsample
                )
                
                # Replace original output with background-removed version
                if os.path.exists(temp_bg_removed):
                    os.replace(temp_bg_removed, output_path)
                    logging.info("✅ Background removal complete")
                
            except Exception as bg_error:
                logging.error(f"Background removal failed: {bg_error}")
                import traceback
                traceback.print_exc()
                logging.warning("Keeping video without background removal")
        
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
        # Cleanup video object if it exists
        try:
            if 'video' in locals():
                video.close()
                logging.debug("Closed video object")
        except Exception as e:
            logging.warning(f"Error closing video object: {e}")
        
        # Cleanup temporary audio
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
