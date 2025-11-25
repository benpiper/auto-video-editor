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

    args = parser.parse_args()

    logging.info(f"Processing {args.input_file} -> {args.output_file}")
    process_video(
        args.input_file,
        args.output_file,
        args.min_silence,
        args.silence_thresh,
        args.crossfade,
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
