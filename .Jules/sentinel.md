## 2026-08-20 - Werkzeug Interactive Debugger Exposure
**Vulnerability:** Flask application bound to 0.0.0.0 with debug=True enabled, exposing the interactive debugger.
**Learning:** Running Flask with debug mode on a public interface allows unauthenticated remote code execution via the Werkzeug debugger.
**Prevention:** Ensure debug=False is explicitly set when binding to public IP addresses (0.0.0.0), or prefer using a production WSGI server instead of the built-in development server.
