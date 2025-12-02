import os
import logging
import subprocess
import tempfile
from typing import Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB tuple (0-255 range).
    
    Args:
        hex_color: Hex color string (e.g., '#00FF00' or '00FF00')
    
    Returns:
        RGB tuple with values in 0-255 range
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def parse_background_color(color: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse background color from string.
    
    Args:
        color: Color string ('transparent', hex code, or named color)
    
    Returns:
        RGB tuple (0-255) or None for transparent
    """
    if color.lower() == 'transparent':
        return None
    
    # Named colors (BGR format for OpenCV)
    named_colors = {
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (0, 0, 255),
    }
    
    if color.lower() in named_colors:
        return named_colors[color.lower()]
    
    # Try parsing as hex
    if color.startswith('#') or len(color) == 6:
        try:
            rgb = hex_to_rgb(color)
            # Convert to BGR for OpenCV
            return (rgb[2], rgb[1], rgb[0])
        except ValueError:
            logging.warning(f"Invalid hex color: {color}, defaulting to green")
            return (0, 255, 0)
    
    logging.warning(f"Unknown color: {color}, defaulting to green")
    return (0, 255, 0)


def load_background_image(image_path: str, width: int, height: int) -> np.ndarray:
    """
    Load and prepare background image for compositing.
    
    Args:
        image_path: Path to background image file
        width: Target width (video width)
        height: Target height (video height)
    
    Returns:
        Background image as numpy array (H, W, 3) in BGR format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Background image not found: {image_path}")
    
    logging.info(f"Loading background image: {image_path}")
    
    # Load image with OpenCV (BGR format)
    bg_image = cv2.imread(image_path)
    if bg_image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Get original dimensions
    orig_height, orig_width = bg_image.shape[:2]
    logging.info(f"Background image original size: {orig_width}x{orig_height}")
    
    # Resize to match video dimensions
    if orig_width != width or orig_height != height:
        bg_image = cv2.resize(bg_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        logging.info(f"Resized background image to: {width}x{height}")
    
    return bg_image


def segment_person_from_video(
    input_path: str,
    output_path: str,
    model_type: str = 'general',
    background_color: str = 'green',
    background_image: Optional[str] = None,
    threshold: float = 0.5,
    smooth_radius: int = 5
) -> str:
    """
    Remove background from video using MediaPipe person segmentation.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        model_type: Segmentation model ('general' for multi-person, 'landscape' for portrait)
        background_color: Background color or 'transparent'
        background_image: Path to background image file (takes precedence over background_color)
        threshold: Confidence threshold for person detection (0.0-1.0)
        smooth_radius: Mask smoothing radius in pixels (0 to disable)
    
    Returns:
        Path to output video
    """
    logging.info(f"Removing background from {input_path} using person segmentation...")
    
    # Initialize MediaPipe segmentation
    # Model selection: 0 = general (landscape videos), 1 = landscape (portrait/selfie)
    model_selection = 0 if model_type == 'general' else 1
    
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection)
    
    # Read video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    logging.info(f"Model type: {model_type} (model_selection={model_selection})")
    logging.info(f"Confidence threshold: {threshold}")
    logging.info(f"Smoothing radius: {smooth_radius}")
    
    # Parse background
    bg_color = None
    bg_image = None
    
    if background_image:
        bg_image = load_background_image(background_image, width, height)
        logging.info("Using background image")
    else:
        bg_color = parse_background_color(background_color)
        if bg_color is not None:
            logging.info(f"Using background color: {background_color}")
        else:
            logging.info("Using transparent background")
    
    # Determine output codec
    if bg_color is None and bg_image is None:  # Transparent
        fourcc = cv2.VideoWriter_fourcc(*'png ')
        if not output_path.endswith('.mov') and not output_path.endswith('.avi'):
            output_path_temp = output_path.rsplit('.', 1)[0] + '.mov'
            logging.info(f"Changed output to {output_path_temp} for alpha channel support")
        else:
            output_path_temp = output_path
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path_temp = output_path
    
    # Create video writer
    out = cv2.VideoWriter(
        output_path_temp,
        fourcc,
        fps,
        (width, height)
    )
    
    frame_count = 0
    logging.info("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            logging.info(f"Processed {frame_count}/{total_frames} frames ({frame_count*100//total_frames}%)")
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = segmenter.process(image_rgb)
        
        # Get segmentation mask
        # results.segmentation_mask is a float array with values 0-1
        # where higher values indicate person presence
        condition = results.segmentation_mask > threshold
        
        # Apply smoothing to reduce flickering
        if smooth_radius > 0:
            # Convert boolean mask to uint8 for blur operation
            mask_uint8 = condition.astype(np.uint8) * 255
            # Apply Gaussian blur
            blurred_mask = cv2.GaussianBlur(mask_uint8, (smooth_radius*2+1, smooth_radius*2+1), 0)
            # Convert back to float 0-1 for compositing
            mask = blurred_mask.astype(np.float32) / 255.0
            # Expand to 3 channels
            mask_3ch = np.stack([mask] * 3, axis=-1)
        else:
            # No smoothing, just use threshold
            mask_3ch = np.stack([condition] * 3, axis=-1).astype(np.float32)
        
        # Composite the output
        if bg_color is None and bg_image is None:
            # Transparent background - create RGBA
            # Extract person (keep only foreground)
            output_rgb = (image_rgb * mask_3ch).astype(np.uint8)
            # Convert mask to alpha channel
            alpha = (mask_3ch[:, :, 0] * 255).astype(np.uint8)
            # Create RGBA
            output_rgba = np.dstack([output_rgb, alpha])
            # Convert to BGR for OpenCV
            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            output_bgra = np.dstack([output_bgr, alpha])
            out.write(output_bgra)
        elif bg_image is not None:
            # Composite with background image
            # foreground * mask + background * (1 - mask)
            output_rgb = (image_rgb * mask_3ch + bg_image[:, :, ::-1] * (1 - mask_3ch)).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            out.write(output_bgr)
        else:
            # Solid background color
            # Create background with the same dimensions
            bg_frame = np.full_like(frame, bg_color, dtype=np.uint8)
            # Composite: foreground * mask + background * (1 - mask)
            output = (frame * mask_3ch + bg_frame * (1 - mask_3ch)).astype(np.uint8)
            out.write(output)
    
    cap.release()
    out.release()
    segmenter.close()
    
    # Merge audio from original video using FFmpeg
    logging.info("Merging audio from original video...")
    
    # Create temporary file for the final output with audio
    temp_fd, temp_with_audio = tempfile.mkstemp(suffix='.mp4', prefix='seg_audio_')
    os.close(temp_fd)
    
    try:
        cmd = [
            'ffmpeg',
            '-i', output_path_temp,  # Video without audio
            '-i', input_path,  # Original video with audio
            '-map', '0:v:0',  # Take video from first input
            '-map', '1:a:0?',  # Take audio from second input (optional)
            '-c', 'copy',  # Copy streams without re-encoding
            '-shortest',  # Match shortest stream duration
            '-y',  # Overwrite output
            temp_with_audio
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            # Replace the video-only output with the version that has audio
            os.replace(temp_with_audio, output_path_temp)
            logging.info("Audio merged successfully")
        else:
            logging.warning(f"Could not merge audio (video may have no audio track): {result.stderr}")
            if os.path.exists(temp_with_audio):
                os.remove(temp_with_audio)
    
    except Exception as e:
        logging.warning(f"Error merging audio: {e}")
        if os.path.exists(temp_with_audio):
            os.remove(temp_with_audio)
    
    logging.info(f"Person segmentation complete: {output_path_temp}")
    return output_path_temp


if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) >= 3:
        segment_person_from_video(
            sys.argv[1],
            sys.argv[2],
            background_color=sys.argv[3] if len(sys.argv) > 3 else 'green'
        )
    else:
        print("Usage: python person_segmenter.py input.mp4 output.mp4 [background_color]")
