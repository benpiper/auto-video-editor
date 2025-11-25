#!/usr/bin/env python3
"""
Auto Video Editor - Web Interface
Simple web frontend for video processing
"""

if __name__ == '__main__':
    import os
    import sys
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Import and run the Flask app
    from app import app
    
    print("=" * 60)
    print("Auto Video Editor - Web Interface")
    print("=" * 60)
    print(f"Starting server on http://localhost:5000")
    print("Press CTRL+C to stop")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
