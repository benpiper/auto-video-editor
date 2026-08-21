## 2026-08-21 - Werkzeug Interactive Debugger Exposure
**Vulnerability:** The Flask application was binding to public interfaces (`0.0.0.0`) while running with `debug=True` in `web_app/app.py` and `web_app/run.py`.
**Learning:** Exposing the Werkzeug interactive debugger on public interfaces is a critical security vulnerability, as it allows arbitrary remote code execution on the server.
**Prevention:** Always ensure development configurations (e.g., `debug=True`) are disabled or appropriately secured before allowing connections from all IP addresses (`0.0.0.0`), especially in environments that mimic or might be used as production.
