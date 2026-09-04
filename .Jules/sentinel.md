## 2026-09-04 - Prevent Werkzeug Interactive Debugger Exposure
**Vulnerability:** Flask application bound to 0.0.0.0 with debug=True enabled, exposing the interactive debugger.
**Learning:** Development configurations (debug=True) must not be active when binding to public interfaces, as this enables remote code execution vulnerabilities via the Werkzeug debugger.
**Prevention:** Always set debug=False in production or when binding to 0.0.0.0. Use environment variables to control debug mode explicitly for local development on 127.0.0.1.
