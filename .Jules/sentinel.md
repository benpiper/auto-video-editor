
## 2026-08-31 - Disable Werkzeug interactive debugger on public interface
**Vulnerability:** Flask application started with `debug=True` and bound to `0.0.0.0`, exposing the Werkzeug interactive debugger and enabling Remote Code Execution (RCE).
**Learning:** Development configurations (like `debug=True`) must never be active when a server binds to public or all interfaces (`0.0.0.0`), as they often contain administrative endpoints or debuggers meant only for local use.
**Prevention:** Always set `debug=False` when binding to `0.0.0.0`, or bind strictly to `127.0.0.1` when using debug mode.
