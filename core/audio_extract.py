import av
import os
import logging
from typing import Optional, Callable

logger = logging.getLogger("AudioExtract")


def extract_audio_pyav(
    video_path: str,
    output_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Extracts audio from video file using PyAV for high efficiency.
    Optimized for Whisper (16kHz, mono).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        container = av.open(video_path)

        # Check if container has audio streams
        if not container.streams.audio:
            logger.warning(f"No audio streams found in {video_path}")
            container.close()
            return False

        input_stream = container.streams.audio[0]

        # Create output container for WAV
        output_container = av.open(output_path, "w", format="wav")
        # Try to set channels/rate via codec_context if add_stream doesn't take them directly or they are read-only
        output_stream = output_container.add_stream("pcm_s16le", rate=sample_rate)
        if channels == 1:
            output_stream.codec_context.layout = "mono"
        else:
            output_stream.codec_context.layout = "stereo"
        output_stream.codec_context.sample_rate = sample_rate

        # Setup Resampler
        resampler = av.AudioResampler(
            format="s16",
            layout="mono" if channels == 1 else "stereo",
            rate=sample_rate,
        )

        total_frames = input_stream.frames if input_stream.frames > 0 else None
        processed_frames = 0

        for frame in container.decode(audio=0):
            # Resample and encode
            resampled_frames = resampler.resample(frame)
            for resampled_frame in resampled_frames:
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)

            processed_frames += 1
            if total_frames and progress_callback and processed_frames % 100 == 0:
                progress = min(processed_frames / total_frames, 1.0)
                progress_callback(progress, f"Extracting audio: {int(progress * 100)}%")

        # Flush encoder
        for packet in output_stream.encode(None):
            output_container.mux(packet)

        output_container.close()
        container.close()
        logger.info(f"Successfully extracted audio to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to extract audio with PyAV: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        return False
