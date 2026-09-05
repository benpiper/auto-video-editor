## 2026-09-05 - Prevent Werkzeug RCE on 0.0.0.0
**Vulnerability:** Running Flask with `debug=True` on the public interface (0.0.0.0) exposes the Werkzeug interactive debugger.
**Learning:** The interactive debugger allows arbitrary Python code execution. Exposing it on all interfaces creates a critical RCE vulnerability, especially in production environments or public-facing servers.
**Prevention:** Never use `debug=True` when `host='0.0.0.0'`. Use `debug=False` to ensure the debugger is not reachable.
