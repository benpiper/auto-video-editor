import shutil
import sys
import os
import logging

# Logger initialization (without global basicConfig to avoid hijacking logs)
logger = logging.getLogger("BaseWorker")


class BaseWorker:
    """Base class for all background workers in the Auto Video Editor."""

    @staticmethod
    def verify_environment() -> None:
        """
        Verifies that the execution environment meets hardware and storage requirements.
        Specifically checks for /dev/shm capacity.
        """
        ramdisk_path = "/dev/shm"

        # Check if /dev/shm exists
        if not os.path.exists(ramdisk_path):
            logger.error(f"RAMDisk path {ramdisk_path} does not exist.")
            sys.exit(1)

        # Check free space
        try:
            total, used, free = shutil.disk_usage(ramdisk_path)
            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            free_gb = free / (1024**3)
            logger.info(
                f"RAMDisk {ramdisk_path} Status: Total={total_gb:.2f}GB, "
                f"Used={used_gb:.2f}GB, Free={free_gb:.2f}GB"
            )

            if free < 2 * 1024**3:  # < 2GB
                logger.error(
                    f"Insufficient RAMDisk space: {free_gb:.2f} GB (minimum 2GB required)"
                )
                sys.exit(1)

        except Exception as e:
            logger.error(f"Failed to check RAMDisk usage: {e}")
            sys.exit(1)


if __name__ == "__main__":
    # Configure logging only when run as main
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Test verify_environment if run directly
    BaseWorker.verify_environment()
    print("Environment verified successfully.")
