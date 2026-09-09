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
    
    # 🛡️ Sentinel: Ensure debug mode is disabled when binding to 0.0.0.0 to prevent RCE
    is_debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    bind_host = '127.0.0.1' if is_debug else '0.0.0.0'
    app.run(debug=is_debug, host=bind_host, port=5000, threaded=True, use_reloader=False)
