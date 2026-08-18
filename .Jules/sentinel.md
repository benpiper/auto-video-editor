## 2026-08-18 - Werkzeug Debugger RCE Prevention
**Vulnerability:** Flask application bound to public interface (`0.0.0.0`) with `debug=True`, exposing Werkzeug interactive debugger which allows arbitrary code execution.
**Learning:** Development configurations often leak into production-like binding environments. The debugger enables unauthenticated remote code execution on the host machine.
**Prevention:** Always set `debug=False` when binding to `0.0.0.0` or any external interface, and explicitly manage environment-specific configurations instead of hardcoding `debug=True`.
