from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import os
import sys
import uuid
import threading
import json
import time
from datetime import datetime

# Add parent directory to path to import processor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processor import process_video

app = Flask(__name__)

# Use absolute paths for upload/output folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB max

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# In-memory job tracking
jobs = {}

class Job:
    def __init__(self, job_id, filename):
        self.job_id = job_id
        self.filename = filename
        self.status = 'pending'
        self.progress = 0
        self.message = 'Waiting to start...'
        self.created_at = datetime.now()
        self.output_path = None
        self.error = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save file
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    file.save(input_path)
    
    # Get parameters
    params = {
        'min_silence': int(request.form.get('min_silence', 2000)),
        'silence_thresh': int(request.form.get('silence_thresh', -63)),
        'crossfade': float(request.form.get('crossfade', 0.2)),
        'bitrate': request.form.get('bitrate', '5000k'),
        'preset': request.form.get('preset', 'medium'),
        'use_crf': request.form.get('use_crf') == 'true',
        'crf': int(request.form.get('crf', 18)),
        'filler_words': [w.strip() for w in request.form.get('filler_words', '').split(';') if w.strip()]
    }
    
    # Create job
    job = Job(job_id, filename)
    jobs[job_id] = job
    
    # Start processing in background thread
    # Generate output filename: {original_name}_edited_{YYYYMMDD_HHMM}.mp4
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}_edited_{timestamp}.mp4"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    thread = threading.Thread(
        target=process_video_async,
        args=(job_id, input_path, output_path, params)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id})

def process_video_async(job_id, input_path, output_path, params):
    """Process video in background thread"""
    job = jobs[job_id]
    
    try:
        job.status = 'processing'
        job.message = 'Starting video processing...'
        job.progress = 5
        
        # Process video
        process_video(
            input_path,
            output_path,
            params['min_silence'],
            params['silence_thresh'],
            0.2,    # crossfade_duration (ignored with no_crossfade=True)
            "5000k", # bitrate (ignored, auto-detected)
            18,     # crf (ignored)
            "medium", # preset (ignored)
            False,  # use_crf
            False,  # use_gpu_encoding
            True,   # no_crossfade (FORCE TRUE as requested)
            params['filler_words']  # custom filler words
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
            os.remove(input_path)

@app.route('/progress/<job_id>')
def progress(job_id):
    """Server-Sent Events endpoint for progress updates"""
    def generate():
        if job_id not in jobs:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return
        
        job = jobs[job_id]
        last_progress = -1
        
        while True:
            if job.progress != last_progress or job.status in ['complete', 'error']:
                data = {
                    'status': job.status,
                    'progress': job.progress,
                    'message': job.message
                }
                yield f"data: {json.dumps(data)}\n\n"
                last_progress = job.progress
                
                if job.status in ['complete', 'error']:
                    break
            
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<job_id>')
def download(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job.status != 'complete' or not job.output_path:
        return jsonify({'error': 'Video not ready'}), 400
    
    return send_file(job.output_path, as_attachment=True, download_name='edited_video.mp4')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
