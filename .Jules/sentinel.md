
## 2026-08-30 - Prevent Werkzeug Debugger RCE
**Vulnerability:** Flask application bound to public interface (0.0.0.0) with `debug=True`, exposing the interactive Werkzeug debugger.
**Learning:** Development configurations active on public IP bindings allow attackers to execute arbitrary Python code remotely.
**Prevention:** Ensure debug mode is explicitly set to False when binding to 0.0.0.0 in application entry points.
