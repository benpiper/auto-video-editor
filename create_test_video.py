import numpy as np
import moviepy.editor as mp
from moviepy.audio.AudioClip import AudioArrayClip
import os


def create_tone(duration, freq=440, volume=0.5, fps=44100):
    """Creates a sine wave tone."""
    t = np.linspace(0, duration, int(fps * duration))
    # reshape to (N, 1) for mono or (N, 2) for stereo. MoviePy expects (N, nchannels)
    audio = volume * np.sin(2 * np.pi * freq * t)
    return audio.reshape(-1, 1)


def create_test_video(output_path="test_video.mp4"):
    print("Generating test video...")
    fps = 24
    audio_fps = 44100

    # Create audio segments
    # 1. Speech (Tone A) - 3s
    # 2. Silence - 3s (Should be removed)
    # 3. Speech (Tone B) - 3s
    # 4. Short Silence - 0.5s (Should be kept)
    # 5. Speech (Tone C) - 3s

    tone_a = create_tone(3.0, 440)
    silence_long = np.zeros((int(3.0 * audio_fps), 1))
    tone_b = create_tone(3.0, 880)
    silence_short = np.zeros((int(0.5 * audio_fps), 1))
    tone_c = create_tone(3.0, 660)

    full_audio_arr = np.concatenate(
        [tone_a, silence_long, tone_b, silence_short, tone_c]
    )
    audio_clip = AudioArrayClip(full_audio_arr, fps=audio_fps)

    duration = audio_clip.duration

    # Create a visual clip (just a color screen with text maybe, or changing colors)
    # Let's make it change colors to visualize cuts

    clip1 = mp.ColorClip(size=(640, 480), color=(255, 0, 0), duration=3.0)
    clip2 = mp.ColorClip(
        size=(640, 480), color=(0, 0, 0), duration=3.0
    )  # Black during silence
    clip3 = mp.ColorClip(size=(640, 480), color=(0, 255, 0), duration=3.0)
    clip4 = mp.ColorClip(
        size=(640, 480), color=(0, 0, 0), duration=0.5
    )  # Black during short silence
    clip5 = mp.ColorClip(size=(640, 480), color=(0, 0, 255), duration=3.0)

    video_clip = mp.concatenate_videoclips([clip1, clip2, clip3, clip4, clip5])
    video_clip = video_clip.set_audio(audio_clip)

    video_clip.write_videofile(output_path, fps=fps)
    print(f"Test video created at {output_path}")


if __name__ == "__main__":
    create_test_video()
