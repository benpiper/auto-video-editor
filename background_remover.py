import os
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
import torch
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model URLs
MODEL_URLS = {
    'mobilenetv3': 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth',
    'resnet50': 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth',
}

# Cache directory for models
CACHE_DIR = Path.home() / '.cache' / 'auto-video-editor' / 'rvm_models'


def download_model(model_name: str = 'mobilenetv3') -> Path:
    """
    Download RVM model if not already cached.
    
    Args:
        model_name: Model variant to download ('mobilenetv3' or 'resnet50')
    
    Returns:
        Path to the downloaded model file
    """
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}. Choose 'mobilenetv3' or 'resnet50'")
    
    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = CACHE_DIR / f'rvm_{model_name}.pth'
    
    # Download if not exists
    if not model_path.exists():
        logging.info(f"Downloading RVM model '{model_name}' (~140MB)...")
        url = MODEL_URLS[model_name]
        
        try:
            # Download with progress
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                if count * block_size < total_size:
                    print(f"\rDownloading: {percent}%", end='', flush=True)
                else:
                    print(f"\rDownloading: 100%")
            
            urllib.request.urlretrieve(url, model_path, reporthook=reporthook)
            logging.info(f"Model downloaded to {model_path}")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            raise
    else:
        logging.info(f"Using cached model: {model_path}")
    
    return model_path


def load_rvm_model(model_name: str = 'mobilenetv3', device: str = 'auto') -> torch.nn.Module:
    """
    Load RVM model.
    
    Args:
        model_name: Model variant ('mobilenetv3' or 'resnet50')
        device: Device to load model on ('cuda', 'cpu', or 'auto')
    
    Returns:
        Loaded RVM model
    """
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Loading RVM model '{model_name}' on {device}...")
    
    # Download model if needed
    model_path = download_model(model_name)
    
    # Import the model architecture
    # We'll use torch.hub to load the model architecture
    try:
        model = torch.hub.load("PeterL1n/RobustVideoMatting", model_name, trust_repo=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.eval().to(device)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex color to RGB tuple (0-1 range).
    
    Args:
        hex_color: Hex color string (e.g., '#00FF00' or '00FF00')
    
    Returns:
        RGB tuple with values in 0-1 range
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def parse_background_color(color: str) -> Optional[Tuple[float, float, float]]:
    """
    Parse background color from string.
    
    Args:
        color: Color string ('transparent', hex code, or named color)
    
    Returns:
        RGB tuple or None for transparent
    """
    if color.lower() == 'transparent':
        return None
    
    # Named colors
    named_colors = {
        'green': (0.0, 1.0, 0.0),
        'blue': (0.0, 0.0, 1.0),
        'white': (1.0, 1.0, 1.0),
        'black': (0.0, 0.0, 0.0),
        'red': (1.0, 0.0, 0.0),
    }
    
    if color.lower() in named_colors:
        return named_colors[color.lower()]
    
    # Try parsing as hex
    if color.startswith('#') or len(color) == 6:
        try:
            return hex_to_rgb(color)
        except ValueError:
            logging.warning(f"Invalid hex color: {color}, defaulting to green")
            return (0.0, 1.0, 0.0)
    
    logging.warning(f"Unknown color: {color}, defaulting to green")
    return (0.0, 1.0, 0.0)


def load_background_image(image_path: str, width: int, height: int, device: str = 'cpu') -> torch.Tensor:
    """
    Load and prepare background image for compositing.
    
    Args:
        image_path: Path to background image file
        width: Target width (video width)
        height: Target height (video height)
        device: Device to load tensor on
    
    Returns:
        Background image as tensor (3, H, W) normalized to 0-1, in RGB order
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
    
    # Convert BGR to RGB
    bg_image_rgb = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize to 0-1
    bg_tensor = torch.from_numpy(bg_image_rgb).permute(2, 0, 1).float().div(255.0).to(device)
    
    return bg_tensor


def remove_background(
    input_path: str,
    output_path: str,
    model_name: str = 'mobilenetv3',
    background_color: str = 'green',
    background_image: Optional[str] = None,
    downsample_ratio: Optional[float] = None,
    device: str = 'auto'
) -> None:
    """
    Remove background from video using RVM.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        model_name: RVM model variant
        background_color: Background color or 'transparent' (ignored if background_image is set)
        background_image: Path to background image file (takes precedence over background_color)
        downsample_ratio: Downsample ratio for processing (None for auto)
        device: Device to use for inference
    """
    logging.info(f"Removing background from {input_path}...")
    
    # Load model
    model = load_rvm_model(model_name, device)
    
    # Parse background color
    bg_color = parse_background_color(background_color)
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Read video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Auto-determine downsample ratio
    if downsample_ratio is None:
        if width * height > 1920 * 1080:  # 4K
            downsample_ratio = 0.125
        elif width * height > 1280 * 720:  # HD
            downsample_ratio = 0.25
        else:
            downsample_ratio = 0.5
        logging.info(f"Auto-selected downsample_ratio: {downsample_ratio}")
    
    # Determine output codec and format
    if bg_color is None:  # Transparent
        # Use a codec that supports alpha channel
        fourcc = cv2.VideoWriter_fourcc(*'png ')  # PNG codec
        output_path_temp = output_path
        if not output_path.endswith('.mov') and not output_path.endswith('.avi'):
            output_path_temp = output_path.rsplit('.', 1)[0] + '.mov'
            logging.info(f"Changed output to {output_path_temp} for alpha channel support")
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create video writer
    out = cv2.VideoWriter(
        output_path if bg_color is not None else output_path_temp,
        fourcc,
        fps,
        (width, height)
    )
    
    # Recurrent states for temporal consistency
    rec = [None] * 4
    
    # Prepare background (image takes precedence over color)
    bg_tensor = None
    bg_color = None
    
    if background_image:
        # Load background image
        bg_tensor = load_background_image(background_image, width, height, device)
        logging.info("Using background image")
    else:
        # Parse background color
        bg_color = parse_background_color(background_color)
        if bg_color is not None:
            # Convert background color to tensor
            bgr_tensor = torch.tensor([bg_color[2], bg_color[1], bg_color[0]]).view(3, 1, 1).to(device)
            logging.info(f"Using background color: {background_color}")
    
    frame_count = 0
    logging.info("Processing frames...")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                logging.info(f"Processed {frame_count}/{total_frames} frames ({frame_count*100//total_frames}%)")
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and normalize to 0-1
            src = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            
            # Run model
            fgr, pha, *rec = model(src, *rec, downsample_ratio)
            
            # Compose output
            if bg_color is None and bg_tensor is None:
                # Create RGBA output (transparent background)
                fgr_np = fgr[0].cpu().numpy()
                pha_np = pha[0].cpu().numpy()
                
                # Convert to 0-255 range
                fgr_np = (fgr_np * 255).astype(np.uint8)
                pha_np = (pha_np * 255).astype(np.uint8)
                
                # Create RGBA
                rgba = np.dstack((fgr_np[2], fgr_np[1], fgr_np[0], pha_np[0]))
                out.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            elif bg_tensor is not None:
                # Composite with background image
                com = fgr * pha + bg_tensor.unsqueeze(0) * (1 - pha)
                
                # Convert back to numpy and BGR
                com_np = com[0].cpu().numpy()
                com_np = (com_np * 255).astype(np.uint8)
                com_bgr = np.transpose(com_np, (1, 2, 0))[:, :, ::-1]
                
                out.write(com_bgr)
            else:
                # Composite with solid background color
                com = fgr * pha + bgr_tensor * (1 - pha)
                
                # Convert back to numpy and BGR
                com_np = com[0].cpu().numpy()
                com_np = (com_np * 255).astype(np.uint8)
                com_bgr = np.transpose(com_np, (1, 2, 0))[:, :, ::-1]
                
                out.write(com_bgr)
    
    cap.release()
    out.release()
    
    # Merge audio from original video using FFmpeg
    # OpenCV VideoWriter doesn't preserve audio, so we need to add it back
    logging.info("Merging audio from original video...")
    
    import subprocess
    import tempfile
    
    # Create temporary file for the final output with audio
    temp_fd, temp_with_audio = tempfile.mkstemp(suffix='.mp4', prefix='bg_removed_audio_')
    os.close(temp_fd)
    
    try:
        # Use FFmpeg to copy audio from input to output
        # -i output without audio, -i original input with audio
        # -map 0:v takes video from first input, -map 1:a takes audio from second input
        # -c copy avoids re-encoding (fast)
        cmd = [
            'ffmpeg',
            '-i', output_path if bg_color is not None else output_path_temp,  # Video without audio
            '-i', input_path,  # Original video with audio
            '-map', '0:v:0',  # Take video from first input
            '-map', '1:a:0?',  # Take audio from second input (? makes it optional)
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
            final_output = output_path if bg_color is not None else output_path_temp
            os.replace(temp_with_audio, final_output)
            logging.info(f"Audio merged successfully")
        else:
            logging.warning(f"Could not merge audio (video may have no audio track): {result.stderr}")
            # Clean up temp file
            if os.path.exists(temp_with_audio):
                os.remove(temp_with_audio)
    
    except Exception as e:
        logging.warning(f"Error merging audio: {e}")
        if os.path.exists(temp_with_audio):
            os.remove(temp_with_audio)
    
    logging.info(f"Background removal complete: {output_path}")


def apply_background_removal(
    input_path: str,
    output_path: str,
    model_name: str = 'mobilenetv3',
    background_color: str = 'green',
    background_image: Optional[str] = None,
    downsample_ratio: Optional[float] = None
) -> str:
    """
    Main function to apply background removal to a video file.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        model_name: RVM model variant ('mobilenetv3' or 'resnet50')
        background_color: Background color or 'transparent' (ignored if background_image is set)
        background_image: Path to background image file
        downsample_ratio: Downsample ratio (None for auto)
    
    Returns:
        Path to output video
    """
    try:
        remove_background(
            input_path,
            output_path,
            model_name,
            background_color,
            background_image,
            downsample_ratio
        )
        return output_path
    except Exception as e:
        logging.error(f"Background removal failed: {e}")
        raise


if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) >= 3:
        apply_background_removal(
            sys.argv[1],
            sys.argv[2],
            background_color=sys.argv[3] if len(sys.argv) > 3 else 'green'
        )
    else:
        print("Usage: python background_remover.py input.mp4 output.mp4 [background_color]")
