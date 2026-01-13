import av
import os
import logging

logger = logging.getLogger("MediaInfo")


def get_video_metadata(filepath: str) -> dict:
    """
    Extracts metadata from a video file using PyAV.
    Returns a dictionary of technical specs.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        container = av.open(filepath)
    except Exception as e:
        logger.error(f"Failed to open media file {filepath}: {e}")
        raise ValueError(f"Invalid or unsupported media file: {e}")

    metadata = {
        "filename": os.path.basename(filepath),
        "path": filepath,
        "format": container.format.name,
        "duration": float(container.duration / av.time_base)
        if container.duration
        else 0,
        "streams": [],
    }

    # Extract Video Stream Data
    video_streams = container.streams.video
    if video_streams:
        v = video_streams[0]
        metadata["video"] = {
            "width": v.width,
            "height": v.height,
            "codec": v.codec_context.name,
            "pix_fmt": v.pix_fmt,
            "fps": float(v.average_rate) if v.average_rate else None,
            "frames": v.frames if v.frames > 0 else None,
            "bit_rate": v.bit_rate,
        }
        # Add to general metadata for easy access
        metadata["width"] = v.width
        metadata["height"] = v.height
        metadata["fps"] = metadata["video"]["fps"]

    # Extract Audio Stream Data
    audio_streams = container.streams.audio
    if audio_streams:
        a = audio_streams[0]
        metadata["audio"] = {
            "codec": a.codec_context.name,
            "channels": a.channels,
            "sample_rate": a.sample_rate,
            "bit_rate": a.bit_rate,
        }

    container.close()
    return metadata


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        import pprint

        pprint.pprint(get_video_metadata(sys.argv[1]))
