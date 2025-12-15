from flask import Blueprint, request, jsonify, current_app, send_file
import os
import uuid
import threading
import sys
from werkzeug.utils import secure_filename
from datetime import datetime
from .state import jobs, Job

# Import processor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processor import process_video

api_bp = Blueprint('api', __name__, url_prefix='/api')

def process_video_async(job_id, input_path, output_path, params):
    """Background worker for API jobs"""
    job = jobs[job_id]
    
    try:
        job.status = 'processing'
        job.message = 'Starting video processing...'
        job.progress = 5
        
        def update_progress(progress, message):
            job.progress = progress
            job.message = message

        # Extract parameters with defaults
        process_video(
            input_path,
            output_path,
            params.get('min_silence', 2000),
            params.get('silence_thresh', -63),
            params.get('crossfade', 0.2),
            params.get('bitrate', '5000k'),
            params.get('crf', 18),
            params.get('preset', 'medium'),
            params.get('use_crf', False),
            params.get('use_gpu_encoding', False),
            params.get('no_crossfade', False),
            params.get('filler_words', []),
            params.get('freeze_duration', None),
            params.get('freeze_noise', 0.001),
            params.get('remove_background', False),
            params.get('bg_color', 'green'),
            params.get('bg_image', None),
            params.get('rvm_model', 'mobilenetv3'),
            params.get('rvm_downsample', None),
            params.get('use_segmentation', False),
            params.get('seg_model', 'general'),
            params.get('seg_threshold', 0.5),
            params.get('seg_smooth', 5),
            params.get('rvm_erode', 0),
            params.get('rvm_dilate', 0),
            params.get('rvm_median', 0),
            params.get('rvm_blur', 0),
            update_progress
        )
        
        job.status = 'complete'
        job.progress = 100
        job.message = 'Processing complete!'
        job.output_path = output_path
        
    except Exception as e:
        job.status = 'error'
        job.error = str(e)
        job.message = f'Error: {str(e)}'
    
    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass
        # Cleanup background image if used and temporary
        if params.get('bg_image') and os.path.exists(params['bg_image']) and 'uploads' in params['bg_image']:
             try:
                os.remove(params['bg_image'])
             except:
                pass

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload a file to be processed"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = secure_filename(file.filename)
    # Use a UUID to prevent collisions and act as a temporary handle
    file_id = str(uuid.uuid4())
    temp_filename = f"{file_id}_{filename}"
    
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    
    file_path = os.path.join(upload_folder, temp_filename)
    file.save(file_path)
    
    return jsonify({
        'message': 'File uploaded successfully',
        'file_id': file_id,
        'filename': temp_filename,
        'original_filename': filename
    })

@api_bp.route('/jobs', methods=['POST'])
def create_job():
    """Create a processing job using JSON parameters"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
        
    data = request.json
    
    # Identify input file
    input_filename = data.get('filename')
    if not input_filename:
        return jsonify({'error': 'filename is required (from /upload)'}), 400
        
    upload_folder = current_app.config['UPLOAD_FOLDER']
    input_path = os.path.join(upload_folder, input_filename)
    
    if not os.path.exists(input_path):
        return jsonify({'error': 'File not found. Please upload first.'}), 404
        
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Basic params
    filename = data.get('original_filename', input_filename)
    
    # Handle background image - simplified for API: assume path or previously uploaded
    # For now, let's assume 'bg_image_path' if locally available, or they need to upload it separately?
    # Let's keep it simple: allow absolute path for bg_image or uploaded filename
    bg_image = data.get('bg_image')
    if bg_image and not os.path.isabs(bg_image):
        # Check if it exists in uploads
        bg_check = os.path.join(upload_folder, bg_image)
        if os.path.exists(bg_check):
            bg_image = bg_check
    
    params = {
        'min_silence': data.get('min_silence', 2000),
        'silence_thresh': data.get('silence_thresh', -63),
        'crossfade': data.get('crossfade', 0.2),
        'bitrate': data.get('bitrate', '5000k'),
        'crf': data.get('crf', 18),
        'preset': data.get('preset', 'medium'),
        'use_crf': data.get('use_crf', False),
        'use_gpu_encoding': data.get('use_gpu_encoding', False),
        'no_crossfade': data.get('no_crossfade', False),
        'filler_words': data.get('filler_words', []),
        'freeze_duration': data.get('freeze_duration'),
        'freeze_noise': data.get('freeze_noise', 0.001),
        'remove_background': data.get('remove_background', False),
        'bg_color': data.get('bg_color', 'green'),
        'bg_image': bg_image,
        'rvm_model': data.get('rvm_model', 'mobilenetv3'),
        'rvm_downsample': data.get('rvm_downsample'),
        'use_segmentation': data.get('use_segmentation', False),
        'seg_model': data.get('seg_model', 'general'),
        'seg_threshold': data.get('seg_threshold', 0.5),
        'seg_smooth': data.get('seg_smooth', 5),
        'rvm_erode': data.get('rvm_erode', 0),
        'rvm_dilate': data.get('rvm_dilate', 0),
        'rvm_median': data.get('rvm_median', 0),
        'rvm_blur': data.get('rvm_blur', 0),
    }

    # Create job
    job = Job(job_id, filename)
    jobs[job_id] = job
    
    # Output setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = os.path.splitext(filename)[0]
    short_job_id = job_id[:8]
    output_filename = f"{base_name}_edited_{timestamp}_{short_job_id}.mp4"
    output_path = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
    
    # Start thread
    thread = threading.Thread(
        target=process_video_async,
        args=(job_id, input_path, output_path, params)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'pending',
        'message': 'Job started'
    }), 201

@api_bp.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get job status"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    job = jobs[job_id]
    
    # Add download URL if complete
    resp = job.to_dict()
    if job.status == 'complete':
        # Assuming app is served at root, construct relative URL
        resp['download_url'] = f"/api/jobs/{job_id}/download"
        
    return jsonify(resp)

@api_bp.route('/jobs/<job_id>/download', methods=['GET'])
def download_job_artifact(job_id):
    """Download the result of a job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    job = jobs[job_id]
    if job.status != 'complete' or not job.output_path:
        return jsonify({'error': 'Job not ready'}), 400
        
    
    # Check if delete_after_download is requested via query param
    delete_after = request.args.get('delete_after', 'false').lower() == 'true'

    if not delete_after:
        return send_file(
            job.output_path,
            as_attachment=True,
            download_name=os.path.basename(job.output_path)
        )
    else:
        # Stream the file and then delete
        def generate():
            with open(job.output_path, 'rb') as f:
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
            mimetype='video/mp4',
            headers={
                'Content-Disposition': f'attachment; filename={os.path.basename(job.output_path)}'
            }
        )
