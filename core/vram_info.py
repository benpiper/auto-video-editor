import os
import pynvml
import logging

logger = logging.getLogger("VRAMInfo")


def get_available_vram_gb() -> float:
    """
    Returns the amount of free VRAM in Gigabytes on the primary GPU.
    Falls back to environment variable or 8GB if NVML is unavailable.
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_gb = info.free / (1024**3)
        pynvml.nvmlShutdown()
        return free_gb
    except Exception as e:
        logger.debug(f"NVML not available, using dummy VRAM info: {e}")
        return float(os.getenv("VRAM_DUMMY_FREE_GB", "8.0"))


def select_whisper_model(available_gb: float) -> str:
    """
    Selects the best Whisper model based on available VRAM.
    """
    if available_gb >= 10.0:
        return "large-v3-turbo"
    elif available_gb >= 5.0:
        return "medium"
    elif available_gb >= 2.0:
        return "small"
    else:
        return "base"
