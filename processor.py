import os
import logging
from typing import List, Tuple
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
from pydub import AudioSegment, silence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            logging.info(f"  Silence {i}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
    else:
        logging.info("No silence intervals found.")
    
    return silence_intervals_sec

def detect_filler_words(audio_path: str, model_size: str = "base") -> List[Tuple[float, float]]:
    """
    Detects filler words using Whisper.
    Returns:
        List of (start, end) tuples in seconds.
    """
    logging.info(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    
    logging.info("Transcribing audio for filler word detection...")
    # We use a prompt to encourage transcribing filler words if possible, though Whisper is trained to remove them.
    # Sometimes standard transcription removes them. We can try to rely on word-level timestamps.
    result = model.transcribe(audio_path, word_timestamps=True, initial_prompt="Um, uh, like, you know.")
    
    filler_words = ["um", "uh", "umm", "uhh", "er", "ah"]
    filler_intervals = []

    for segment in result["segments"]:
        for word in segment["words"]:
            text = word["word"].strip().lower().replace(",", "").replace(".", "")
            if text in filler_words:
                start_time = word["start"]
                end_time = word["end"]
                duration = end_time - start_time
                filler_intervals.append((start_time, end_time))
                logging.info(f"  Filler word detected: '{text}' at {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
    
    logging.info(f"Found {len(filler_intervals)} filler words total.")
    return filler_intervals

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

def process_video(input_path: str, output_path: str, min_silence_len: int = 2000, silence_thresh: int = -40, crossfade_duration: float = 0.1, bitrate: str = "5000k", crf: int = 18, preset: str = "medium", use_crf: bool = False):
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    temp_audio_path = "temp_audio.wav"
    
    try:
        # 1. Extract Audio
        extract_audio(input_path, temp_audio_path)
        
        # 2. Detect Silence
        silence_intervals = detect_silence(temp_audio_path, min_silence_len, silence_thresh)
        
        # 3. Detect Filler Words
        filler_intervals = detect_filler_words(temp_audio_path)
        
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
                logging.info(f"  Segment {i}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
            logging.info(f"Total duration to be removed: {total_removed_duration:.2f}s")
        
        # 5. Get Keep Intervals
        video = VideoFileClip(input_path)
        total_duration = video.duration
        keep_intervals = invert_intervals(merged_remove_intervals, total_duration)
        
        logging.info(f"Original video duration: {total_duration:.2f}s")
        if merged_remove_intervals:
            final_duration = sum(end - start for start, end in keep_intervals)
            logging.info(f"Final video duration: {final_duration:.2f}s (removed {total_duration - final_duration:.2f}s)")
        
        if len(keep_intervals) == 1 and keep_intervals[0] == (0.0, total_duration):
            logging.info("No cuts needed.")
            # Choose encoding mode: CRF (quality-based) or constant bitrate
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
            
            video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                preset=preset,
                audio_bitrate="192k",
                ffmpeg_params=ffmpeg_params
            )
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
            # We can add a fade in/out to each clip to smooth the audio/video
            # But that creates a dip to black/silence.
            # A true crossfade requires overlap.
            
            # Let's implement a simple crossfade by overlapping.
            # We need to extend the clips slightly if possible, but we can't extend beyond the cut points (that's the bad part).
            # So we can only crossfade if we accept that we are blending the very edges of the "good" parts.
            
            # Actually, standard "jump cut" removal usually just does hard cuts. 
            # If "smooth fade transition" means cross dissolve, we need to overlap.
            # Let's use `crossfadein` on clips[1:]
            
            if i > 0 and crossfade_duration > 0:
                clip = clip.crossfadein(crossfade_duration)
            
            final_clips.append(clip)
            
        final_video = concatenate_videoclips(final_clips, method="compose", padding=-crossfade_duration if crossfade_duration > 0 else 0)
        
        logging.info(f"Writing output to {output_path}")
        # Choose encoding mode: CRF (quality-based) or constant bitrate
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
        
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            preset=preset,
            audio_bitrate="192k",
            ffmpeg_params=ffmpeg_params
        )
        
        video.close()
        final_video.close()
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    # For testing
    pass
