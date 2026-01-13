from flask import Blueprint, request, jsonify, current_app, send_file
import os
import uuid
import threading
import sys
from werkzeug.utils import secure_filename
from datetime import datetime
from .state import jobs, Job

# Import core utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.media_info import get_video_metadata
from core.db import Project, CutCandidate
from processor import process_video, extract_audio, detect_silence, detect_filler_words

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/projects/ingest", methods=["POST"])
def ingest_project():
    """Register a local video file and extract metadata."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    file_path = data.get("path")

    if not file_path:
        return jsonify({"error": "File path is required"}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 404

    try:
        # Extract metadata
        metadata = get_video_metadata(file_path)

        # Create persistent project record
        project = Project.create(
            name=metadata["filename"],
            input_path=file_path,
            status="ingested",
            metadata=metadata,
        )

        return jsonify(
            {
                "message": "Project ingested successfully",
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "status": project.status,
                    "metadata": project.metadata,
                },
            }
        ), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/projects", methods=["GET"])
def list_projects():
    """List all projects from the database."""
    try:
        projects = Project.select().order_by(Project.created_at.desc())
        return jsonify(
            [
                {
                    "id": p.id,
                    "name": p.name,
                    "input_path": p.input_path,
                    "status": p.status,
                    "created_at": p.created_at.isoformat(),
                    "metadata": p.metadata,
                }
                for p in projects
            ]
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/projects/<project_id>", methods=["DELETE"])
def delete_project(project_id):
    """Delete a project and its associated files."""
    try:
        project = Project.get_by_id(project_id)

        # Cleanup files on disk
        files_to_remove = [project.input_path, project.output_path]
        for path in files_to_remove:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Warning: Failed to delete file {path}: {e}")

        # Delete from database ( CASCADE will handle tasks )
        project.delete_instance(recursive=True)

        return jsonify({"message": "Project and files deleted successfully"}), 200

    except Project.DoesNotExist:
        return jsonify({"error": "Project not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/projects/<project_id>/detect", methods=["POST"])
def run_detection(project_id):
    """Trigger AI detection for silence and filler words."""
    try:
        project = Project.get_by_id(project_id)
        project.status = "detecting"
        project.save()

        # Start background detection thread
        thread = threading.Thread(
            target=async_detect_pipeline,
            args=(project.id, current_app._get_current_object()),
        )
        thread.start()

        return jsonify({"message": "Detection started", "project_id": project.id}), 202

    except Project.DoesNotExist:
        return jsonify({"error": "Project not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def async_detect_pipeline(project_id, app):
    """Background worker for detection only."""
    with app.app_context():
        try:
            from core.redis_client import RedisManager
            import os

            project = Project.get_by_id(project_id)
            redis_mgr = RedisManager()

            # Use RAMDisk for audio
            ramdisk_path = "/dev/shm"
            if os.path.exists(ramdisk_path) and os.access(ramdisk_path, os.W_OK):
                audio_path = os.path.join(ramdisk_path, f"detect_{project.id}.wav")
            else:
                audio_path = f"detect_{project.id}.wav"

            def broadcast(p, m):
                redis_mgr.add_to_stream(
                    "progress:stream",
                    {"project_id": project.id, "progress": p, "message": m},
                )

            # 1. Extraction
            broadcast(5, "Extracting audio...")
            has_audio = extract_audio(
                project.input_path,
                audio_path,
                progress_callback=lambda p, m: broadcast(5 + int(p * 10), m),
            )

            if not has_audio:
                project.status = "error"
                project.save()
                broadcast(100, "Error: No audio found")
                return

            # 2. Silence Detection
            broadcast(20, "Detecting silence...")
            silences = detect_silence(
                audio_path,
                progress_callback=lambda p, m: broadcast(20 + int(p * 20), m),
            )

            # Save Silences
            for start, end in silences:
                CutCandidate.create(
                    project=project, type="silence", start_time=start, end_time=end
                )

            # 3. Filler Detection
            broadcast(40, "Detecting filler words...")
            fillers, transcript = detect_filler_words(
                audio_path, redis_mgr=redis_mgr
            )  # logic handles its own progress logging or we wrap it

            # Save Fillers
            for start, end in fillers:
                CutCandidate.create(
                    project=project, type="filler", start_time=start, end_time=end
                )

            # Update Project
            project.status = "ready"
            if transcript:
                meta = project.metadata
                meta["transcript"] = transcript
                project.metadata = meta
            project.save()

            broadcast(100, "Detection complete! Ready for review.")

            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)

        except Exception as e:
            print(f"Detection failed: {e}")
            try:
                project = Project.get_by_id(project_id)
                project.status = "error"
                project.save()
                redis_mgr.add_to_stream(
                    "progress:stream",
                    {"project_id": project.id, "progress": 0, "message": f"Error: {e}"},
                )
            except Exception:
                pass


@api_bp.route("/projects/<project_id>/cuts", methods=["GET"])
def get_project_cuts(project_id):
    """List all proposed cuts for a project."""
    try:
        project = Project.get_by_id(project_id)
        cuts = (
            CutCandidate.select()
            .where(CutCandidate.project == project)
            .order_by(CutCandidate.start_time)
        )
        return jsonify(
            [
                {
                    "id": c.id,
                    "type": c.type,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "ignored": c.ignored,
                }
                for c in cuts
            ]
        )
    except Project.DoesNotExist:
        return jsonify({"error": "Project not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/cuts/<cut_id>", methods=["PATCH"])
def update_cut(cut_id):
    """Toggle the 'ignored' status of a cut."""
    try:
        data = request.json or {}
        cut = CutCandidate.get_by_id(cut_id)
        if "ignored" in data:
            cut.ignored = bool(data["ignored"])
            cut.save()
        return jsonify({"message": "Cut updated", "id": cut.id, "ignored": cut.ignored})
    except CutCandidate.DoesNotExist:
        return jsonify({"error": "Cut not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/projects/<project_id>/preview", methods=["POST"])
def generate_preview(project_id):
    """Trigger 30s sandbox preview generation."""
    try:
        project = Project.get_by_id(project_id)

        # Start background preview thread
        thread = threading.Thread(
            target=async_preview_pipeline,
            args=(project.id, current_app._get_current_object()),
        )
        thread.start()

        return jsonify(
            {"message": "Preview generation started", "project_id": project.id}
        ), 202

    except Project.DoesNotExist:
        return jsonify({"error": "Project not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def async_preview_pipeline(project_id, app):
    """Background worker for sandbox preview."""
    with app.app_context():
        try:
            from core.redis_client import RedisManager
            from processor import extract_sandbox_segment, assemble_preview_video
            import os

            project = Project.get_by_id(project_id)
            redis_mgr = RedisManager()

            def broadcast(p, m):
                redis_mgr.add_to_stream(
                    "progress:stream",
                    {
                        "project_id": project.id,
                        "progress": p,
                        "message": m,
                        "type": "preview",
                    },
                )

            # 1. Extraction (Story 4.1)
            ramdisk_path = "/dev/shm"
            if not (os.path.exists(ramdisk_path) and os.access(ramdisk_path, os.W_OK)):
                ramdisk_path = os.path.dirname(project.input_path)

            sandbox_path = os.path.join(ramdisk_path, f"sandbox_{project.id}.mp4")

            broadcast(10, "Extracting sandbox segment...")
            success = extract_sandbox_segment(
                project.input_path, sandbox_path, duration=30.0
            )

            if not success:
                broadcast(0, "Error: Sandbox extraction failed")
                return

            # Store sandbox path in project metadata for now
            meta = project.metadata or {}
            meta["sandbox_path"] = sandbox_path
            project.metadata = meta
            project.save()

            broadcast(50, "Sandbox extracted. Assembling preview...")

            # 2. Assembly (Story 4.2)
            # Fetch cuts that are NOT ignored
            cuts = CutCandidate.select().where(
                (CutCandidate.project == project) & (~CutCandidate.ignored)
            )
            removal_intervals = [(c.start_time, c.end_time) for c in cuts]

            preview_output_path = os.path.join(
                ramdisk_path, f"preview_{project.id}.mp4"
            )

            success = assemble_preview_video(
                sandbox_path,  # Using the source sandbox for local assembly
                preview_output_path,
                removal_intervals,
                max_duration=30.0,
                progress_callback=broadcast,
            )

            if success:
                meta = project.metadata or {}
                meta["preview_path"] = preview_output_path
                project.metadata = meta
                project.save()
                broadcast(100, "Preview generation complete!")
            else:
                broadcast(0, "Error: Preview assembly failed")

        except Exception as e:
            print(f"Preview failed: {e}")
            try:
                redis_mgr.add_to_stream(
                    "progress:stream",
                    {
                        "project_id": project.id,
                        "progress": 0,
                        "message": f"Preview error: {e}",
                        "type": "preview",
                    },
                )
            except Exception:
                pass


def async_render_pipeline(project_id, render_preset, app):
    """Background worker for full project render."""
    with app.app_context():
        try:
            from core.redis_client import RedisManager
            from processor import process_video
            import os

            project = Project.get_by_id(project_id)
            redis_mgr = RedisManager()

            def broadcast(p, m):
                redis_mgr.add_to_stream(
                    "progress:stream",
                    {
                        "project_id": project.id,
                        "progress": p,
                        "message": m,
                        "type": "render",
                    },
                )

            broadcast(5, "Initializing master render...")

            # Cleanup old render if exists
            meta = project.metadata or {}
            old_path = meta.get("final_output_path")
            if old_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                    print(f"Removed old render: {old_path}")
                except Exception as e:
                    print(f"Warning: Failed to remove old render {old_path}: {e}")

            # Fetch non-ignored cuts
            cuts = CutCandidate.select().where(
                (CutCandidate.project == project) & (~CutCandidate.ignored)
            )
            forced_remove_intervals = [(c.start_time, c.end_time) for c in cuts]

            output_dir = os.path.dirname(project.input_path)
            output_name = f"final_{project.id}_{render_preset}.mp4"
            output_path = os.path.join(output_dir, output_name)

            # Call process_video with forced intervals and render_preset
            result = process_video(
                project.input_path,
                output_path,
                render_preset=render_preset,
                forced_remove_intervals=forced_remove_intervals,
                progress_callback=lambda p, m: broadcast(p, m),
                redis_mgr=redis_mgr,
            )

            if result and result.get("status") == "complete":
                project.status = "complete"
                meta = project.metadata or {}
                meta["render_status"] = "complete"
                meta["final_output_path"] = output_path
                project.metadata = meta
                project.save()
                broadcast(100, f"Render complete! Saved to {output_name}")
            else:
                project.status = "error"
                meta = project.metadata or {}
                meta["render_status"] = "error"
                project.metadata = meta
                project.save()
                error_msg = result.get("error") if result else "Unknown error"
                broadcast(0, f"Render failed: {error_msg}")

        except Exception as e:
            print(f"Render failed: {e}")
            try:
                redis_mgr.add_to_stream(
                    "progress:stream",
                    {
                        "project_id": project_id,
                        "progress": 0,
                        "message": f"Render error: {e}",
                        "type": "render",
                    },
                )
            except Exception:
                pass


def process_video_async(job_id, input_path, output_path, params):
    """Background worker for API jobs"""
    job = jobs[job_id]

    try:
        job.status = "processing"
        job.message = "Starting video processing..."
        job.progress = 5

        def update_progress(progress, message):
            job.progress = progress
            job.message = message

        # Extract parameters with defaults
        result = process_video(
            input_path,
            output_path,
            params.get("min_silence", 2000),
            params.get("silence_thresh", -63),
            params.get("crossfade", 0.2),
            params.get("bitrate", "5000k"),
            params.get("crf", 18),
            params.get("preset", "medium"),
            params.get("use_crf", False),
            params.get("use_gpu_encoding", False),
            params.get("no_crossfade", False),
            params.get("filler_words", []),
            params.get("freeze_duration", None),
            params.get("freeze_noise", 0.001),
            params.get("remove_background", False),
            params.get("bg_color", "green"),
            params.get("bg_image", None),
            params.get("rvm_model", "mobilenetv3"),
            params.get("rvm_downsample", None),
            params.get("use_segmentation", False),
            params.get("seg_model", "general"),
            params.get("seg_threshold", 0.5),
            params.get("seg_smooth", 5),
            params.get("rvm_erode", 0),
            params.get("rvm_dilate", 0),
            params.get("rvm_median", 0),
            params.get("rvm_blur", 0),
            params.get("render_preset", "speed"),
            update_progress,
        )

        # Helper to extract transcript
        transcript_text = result.get("transcript") if isinstance(result, dict) else None

        status = result.get("status") if isinstance(result, dict) else None

        if status == "skipped":
            job.status = "skipped"
            job.progress = 100
            job.message = "Video already optimized - no changes made."
            job.output_path = None
            job.transcript = transcript_text
        elif status == "error" or result is False:
            job.status = "error"
            job.error = (
                result.get("error", "Processing failed")
                if isinstance(result, dict)
                else "Processing failed"
            )
            job.message = job.error
        else:
            job.status = "complete"
            job.progress = 100
            job.message = "Processing complete!"
            job.output_path = output_path
            job.transcript = transcript_text

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.message = f"Error: {str(e)}"

    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass
        # Cleanup background image if used and temporary
        if (
            params.get("bg_image")
            and os.path.exists(params["bg_image"])
            and "uploads" in params["bg_image"]
        ):
            try:
                os.remove(params["bg_image"])
            except Exception:
                pass


@api_bp.route("/upload", methods=["POST"])
def upload_file():
    """Upload a file to be processed"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    # Use a UUID to prevent collisions and act as a temporary handle
    file_id = str(uuid.uuid4())
    temp_filename = f"{file_id}_{filename}"

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, temp_filename)
    file.save(file_path)

    return jsonify(
        {
            "message": "File uploaded successfully",
            "file_id": file_id,
            "filename": temp_filename,
            "original_filename": filename,
        }
    )


@api_bp.route("/jobs", methods=["POST"])
def create_job():
    """Create a processing job using JSON parameters"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json

    # Identify input file
    input_filename = data.get("filename")
    if not input_filename:
        return jsonify({"error": "filename is required (from /upload)"}), 400

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    input_path = os.path.join(upload_folder, input_filename)

    if not os.path.exists(input_path):
        return jsonify({"error": "File not found. Please upload first."}), 404

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Basic params
    filename = data.get("original_filename", input_filename)

    # Handle background image - simplified for API: assume path or previously uploaded
    # For now, let's assume 'bg_image_path' if locally available, or they need to upload it separately?
    # Let's keep it simple: allow absolute path for bg_image or uploaded filename
    bg_image = data.get("bg_image")
    if bg_image and not os.path.isabs(bg_image):
        # Check if it exists in uploads
        bg_check = os.path.join(upload_folder, bg_image)
        if os.path.exists(bg_check):
            bg_image = bg_check

    params = {
        "min_silence": data.get("min_silence", 2000),
        "silence_thresh": data.get("silence_thresh", -63),
        "crossfade": data.get("crossfade", 0.2),
        "bitrate": data.get("bitrate", "5000k"),
        "crf": data.get("crf", 18),
        "preset": data.get("preset", "medium"),
        "use_crf": data.get("use_crf", False),
        "use_gpu_encoding": data.get("use_gpu_encoding", False),
        "no_crossfade": data.get("no_crossfade", False),
        "filler_words": data.get("filler_words", []),
        "freeze_duration": data.get("freeze_duration"),
        "freeze_noise": data.get("freeze_noise", 0.001),
        "remove_background": data.get("remove_background", False),
        "bg_color": data.get("bg_color", "green"),
        "bg_image": bg_image,
        "rvm_model": data.get("rvm_model", "mobilenetv3"),
        "rvm_downsample": data.get("rvm_downsample"),
        "use_segmentation": data.get("use_segmentation", False),
        "seg_model": data.get("seg_model", "general"),
        "seg_threshold": data.get("seg_threshold", 0.5),
        "seg_smooth": data.get("seg_smooth", 5),
        "rvm_erode": data.get("rvm_erode", 0),
        "rvm_dilate": data.get("rvm_dilate", 0),
        "rvm_median": data.get("rvm_median", 0),
        "rvm_blur": data.get("rvm_blur", 0),
    }

    # Create job
    job = Job(job_id, filename)
    jobs[job_id] = job

    # Output setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = os.path.splitext(filename)[0]
    short_job_id = job_id[:8]
    output_filename = f"{base_name}_edited_{timestamp}_{short_job_id}.mp4"
    output_path = os.path.join(current_app.config["OUTPUT_FOLDER"], output_filename)

    # Start thread
    thread = threading.Thread(
        target=process_video_async, args=(job_id, input_path, output_path, params)
    )
    thread.daemon = True
    thread.start()

    return jsonify(
        {"job_id": job_id, "status": "pending", "message": "Job started"}
    ), 201


@api_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    """Get job status"""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]

    # Add download URL if complete
    resp = job.to_dict()
    if job.status == "complete":
        # Assuming app is served at root, construct relative URL
        resp["download_url"] = f"/api/jobs/{job_id}/download"

    return jsonify(resp)


@api_bp.route("/jobs/<job_id>/download", methods=["GET"])
def download_job_artifact(job_id):
    """Download the result of a job"""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    if job.status != "complete" or not job.output_path:
        return jsonify({"error": "Job not ready"}), 400

    # Check if delete_after_download is requested via query param
    delete_after = request.args.get("delete_after", "false").lower() == "true"

    if not delete_after:
        return send_file(
            job.output_path,
            as_attachment=True,
            download_name=os.path.basename(job.output_path),
        )
    else:
        # Stream the file and then delete
        def generate():
            with open(job.output_path, "rb") as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    yield chunk

            # Post-download cleanup
            try:
                os.remove(job.output_path)
                del jobs[job_id]
            except Exception as e:
                print(f"Error cleaning up job {job_id}: {e}")

        return current_app.response_class(
            generate(),
            mimetype="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(job.output_path)}"
            },
        )


@api_bp.route("/projects/<project_id>/render", methods=["POST"])
def start_render(project_id):
    """Start a full quality render for the project."""
    try:
        project = Project.get_by_id(project_id)

        # Check if rendering is already in progress
        if project.status == "rendering":
            return jsonify({"message": "Render already in progress."}), 409

        # Get preset (default to speed)
        data = request.json or {}
        render_preset = data.get("preset", "speed")

        # Update project status
        project.status = "rendering"
        meta = project.metadata or {}
        meta["render_status"] = "rendering"
        meta["render_preset"] = render_preset
        project.metadata = meta
        project.save()

        # Start background render thread
        thread = threading.Thread(
            target=async_render_pipeline,
            args=(project.id, render_preset, current_app._get_current_object()),
        )
        thread.start()

        return jsonify(
            {"message": "Final render started", "project_id": project.id}
        ), 202

    except Project.DoesNotExist:
        return jsonify({"error": "Project not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/projects/<project_id>/final/download", methods=["GET"])
def download_final_video(project_id):
    """Download the final rendered video."""
    try:
        project = Project.get_by_id(project_id)
        meta = project.metadata or {}
        final_path = meta.get("final_output_path")

        if not final_path or not os.path.exists(final_path):
            return jsonify({"error": "Final render not found"}), 404

        return send_file(
            final_path, as_attachment=True, download_name=os.path.basename(final_path)
        )
    except Project.DoesNotExist:
        return jsonify({"error": "Project not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/projects/<project_id>/preview/stream", methods=["GET"])
def stream_preview(project_id):
    """Serve the generated preview video file."""
    try:
        project = Project.get_by_id(project_id)
        meta = project.metadata or {}
        preview_path = meta.get("preview_path")

        if not preview_path or not os.path.exists(preview_path):
            return jsonify({"error": "Preview not found"}), 404

        return send_file(preview_path, mimetype="video/mp4")
    except Project.DoesNotExist:
        return jsonify({"error": "Project not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
