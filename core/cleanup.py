import os
import logging
from core.db import Project

logger = logging.getLogger("Cleanup")


def cleanup_orphaned_files(upload_folder: str, output_folder: str):
    """
    Scans directories and deletes files not referenced in the Project database.
    """
    try:
        # Get all referenced paths from database
        referenced_paths = set()
        projects = Project.select(Project.input_path, Project.output_path)
        for p in projects:
            if p.input_path:
                referenced_paths.add(os.path.abspath(p.input_path))
            if p.output_path:
                referenced_paths.add(os.path.abspath(p.output_path))

        # Check Uploads
        _cleanup_dir(upload_folder, referenced_paths, "Uploads")

        # Check Outputs
        _cleanup_dir(output_folder, referenced_paths, "Outputs")

    except Exception as e:
        logger.error(f"Orphan cleanup failed: {e}")


def _cleanup_dir(directory: str, referenced_paths: set, label: str):
    if not os.path.exists(directory):
        return

    for filename in os.listdir(directory):
        file_path = os.path.abspath(os.path.join(directory, filename))

        # Don't delete directories or .gitkeep
        if not os.path.isfile(file_path) or filename == ".gitkeep":
            continue

        if file_path not in referenced_paths:
            try:
                os.remove(file_path)
                logger.info(f"Deleted orphaned {label} file: {filename}")
            except Exception as e:
                logger.warning(f"Failed to delete orphaned file {file_path}: {e}")


if __name__ == "__main__":
    # Example usage (adjust paths as needed)
    logging.basicConfig(level=logging.INFO)
    from web_app.app import app

    with app.app_context():
        cleanup_orphaned_files(app.config["UPLOAD_FOLDER"], app.config["OUTPUT_FOLDER"])
