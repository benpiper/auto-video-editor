## 2026-08-26 - Werkzeug Interactive Debugger Exposure
**Vulnerability:** Running Flask with `debug=True` and `host='0.0.0.0'` exposes the Werkzeug interactive debugger to the public interface, allowing potential remote code execution.
**Learning:** Development configurations like `debug=True` should never be active when binding to public interfaces, as this immediately compromises the server.
**Prevention:** Always set `debug=False` or omit the debug parameter when binding to `0.0.0.0`, and rely on environment variables for enabling debug mode locally.
