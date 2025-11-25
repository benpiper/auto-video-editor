# Web Frontend for Auto Video Editor

A simple web interface for the auto-video-editor with drag-and-drop upload and real-time progress tracking.

## Features

- 🎨 Modern dark-mode UI
- 📤 Drag & drop video upload
- ⚙️ Adjustable processing parameters
- 📊 Real-time progress updates (Server-Sent Events)
- 📥 One-click download
- 🚀 No external dependencies (just Flask)

## Quick Start

### 1. Install Dependencies

Flask is already installed if you ran `uv add flask`.

### 2. Run the Web Server

```bash
python web_app/app.py
```

### 3. Access the Interface

Open your browser to: **http://localhost:5000**

## Usage

1. **Upload Video**
   - Drag & drop a video file or click to browse
   - Supported formats: MP4, AVI, MOV, MKV
   - Max file size: 500MB

2. **Configure Parameters**
   - **Min Silence**: Minimum silence duration to remove (ms)
   - **Silence Threshold**: Audio level threshold (dBFS)
   - **Encoding Preset**: Speed vs quality tradeoff
   - **Bitrate**: Video quality (higher = better)
   - **Crossfade**: Transition duration between cuts

3. **Process**
   - Click "Start Processing"
   - Watch real-time progress
   - Processing time depends on video length and preset

4. **Download**
   - Click "Download Edited Video" when complete
   - Process another video or close the browser

## Architecture

- **Backend**: Flask with threading for async processing
- **Progress Updates**: Server-Sent Events (SSE)
- **Job Tracking**: In-memory (jobs lost on restart)
- **Storage**: Local filesystem (uploads/outputs)

## File Structure

```
web_app/
├── app.py              # Flask application
├── static/
│   ├── css/
│   │   └── style.css   # Styling
│   ├── js/
│   │   └── app.js      # Frontend logic
│   ├── uploads/        # Temporary uploads
│   └── outputs/        # Processed videos
└── templates/
    └── index.html      # Main UI
```

## Configuration

Edit `app.py` to change:

```python
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Max upload size
app.config['UPLOAD_FOLDER'] = 'web_app/static/uploads'
app.config['OUTPUT_FOLDER'] = 'web_app/static/outputs'
```

## Cleanup

The web app automatically deletes uploaded files after processing. Output files remain until manually deleted.

To clean up old output files:

```bash
rm web_app/static/outputs/*
```

## Limitations

- **Single server**: Can't scale horizontally
- **No persistence**: Jobs lost on server restart
- **No authentication**: Anyone with access can use it
- **Local storage**: Files stored on server disk

These are acceptable for personal use or small teams!

## Troubleshooting

### Port already in use

```bash
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Upload fails

- Check file size (max 500MB by default)
- Ensure video format is supported
- Check disk space in uploads folder

### Processing stuck

- Check terminal for error messages
- Ensure `processor.py` is working (test with CLI)
- Restart the server

## Production Deployment

For production use, consider:

1. **Use a production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 web_app.app:app
   ```

2. **Add authentication**
3. **Implement file cleanup cron job**
4. **Use cloud storage for uploads/outputs**
5. **Add rate limiting**

## Development

To modify the UI:

- **Styling**: Edit `web_app/static/css/style.css`
- **Frontend logic**: Edit `web_app/static/js/app.js`
- **HTML structure**: Edit `web_app/templates/index.html`
- **Backend**: Edit `web_app/app.py`

Changes to CSS/JS/HTML are reflected immediately (refresh browser).
Changes to Python require server restart.

## License

Same as the main project (MIT).
