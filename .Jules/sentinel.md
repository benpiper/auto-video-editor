
## 2026-09-02 - Prevent RCE from Flask Debug Mode
**Vulnerability:** Flask application bound to public interfaces (0.0.0.0) with `debug=True`, exposing the Werkzeug interactive debugger which allows Remote Code Execution.
**Learning:** Development configurations often accidentally leak into production code when entry points are hastily written or hardcoded without environment variable checks.
**Prevention:** Never use `debug=True` when host is 0.0.0.0. Always load debug configuration from environment variables and default to False.
