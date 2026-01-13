#!/usr/bin/env python3
"""
Auto Video Editor - Web Interface
Simple web frontend for video processing
"""

if __name__ == "__main__":
    import os
    import sys

    # Ensure root is in path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    # Import and run the Flask app
    from web_app.app import app

    print("=" * 60)
    print("Auto Video Editor - Web Interface")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("Press CTRL+C to stop")
    print("=" * 60)

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
