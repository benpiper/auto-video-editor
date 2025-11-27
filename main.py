import argparse
import logging
from processor import process_video


def main():
    parser = argparse.ArgumentParser(
        description="Auto Video Editor: Removes silence and filler words."
    )
    parser.add_argument("input_file", help="Path to the input video file.")
    parser.add_argument("output_file", help="Path to the output video file.")
    parser.add_argument(
        "--min-silence",
        type=int,
        default=2000,
        help="Minimum silence duration in milliseconds (default: 2000).",
    )
    parser.add_argument(
        "--silence-thresh",
        type=int,
        default=-63,
        help="Silence threshold in dBFS (default: -63).",
    )
    parser.add_argument(
        "--crossfade",
        type=float,
        default=0.2,
        help="Crossfade duration in seconds (default: 0.2).",
    )
    parser.add_argument(
        "--bitrate",
        type=str,
        default="5000k",
        help="Video bitrate (default: 5000k). Higher = better quality.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF quality (0-51, lower = better quality, default: 18).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="Encoding preset (default: medium). Slower = better compression.",
    )
    parser.add_argument(
        "--use-crf",
        action="store_true",
        help="Use CRF mode (quality-based) instead of constant bitrate mode.",
    )
    parser.add_argument(
        "--use-gpu-encoding",
        action="store_true",
        help="Use NVIDIA GPU (NVENC) for video encoding (requires FFmpeg with NVENC support).",
    )
    parser.add_argument(
        "--no-crossfade",
        action="store_true",
        help="Disable crossfades for faster processing (uses simple concatenation instead).",
    )
    parser.add_argument(
        "--filler-words",
        type=str,
        default="um;uh;umm;uhh;er;just;you know;like;you know",
        help="Semicolon-separated list of filler words to remove (default: um;uh;umm;uhh;er;just;you know;like;you know).",
    )

    args = parser.parse_args()

    # Parse filler words
    filler_words_list = [w.strip() for w in args.filler_words.split(';') if w.strip()] if args.filler_words else None
    
    logging.info(f"Processing {args.input_file} -> {args.output_file}")
    process_video(
        args.input_file,
        args.output_file,
        args.min_silence,
        args.silence_thresh,
        args.crossfade,
        args.bitrate,
        args.crf,
        args.preset,
        args.use_crf,
        args.use_gpu_encoding,
        args.no_crossfade,
        filler_words_list,
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
