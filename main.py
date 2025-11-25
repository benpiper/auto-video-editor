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
        default=-40,
        help="Silence threshold in dBFS (default: -40).",
    )
    parser.add_argument(
        "--crossfade",
        type=float,
        default=0.1,
        help="Crossfade duration in seconds (default: 0.1).",
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

    args = parser.parse_args()

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
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
