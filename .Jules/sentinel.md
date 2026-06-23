## 2026-06-23 - Prevent RCE via Werkzeug Debugger on Public Interfaces
**Vulnerability:** The Flask application was bound to `0.0.0.0` with `debug=True` in both `app.py` and `run.py`.
**Learning:** Enabling `debug=True` on public interfaces exposes the Werkzeug interactive debugger, which allows arbitrary Python code execution.
**Prevention:** Ensure `debug=False` when binding to public IP addresses (e.g., `0.0.0.0`), especially in production-like entry points.
