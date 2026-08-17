## 2026-08-17 - Prevent Werkzeug Debugger RCE
**Vulnerability:** Werkzeug interactive debugger was exposed via `app.run(debug=True, host='0.0.0.0')`.
**Learning:** Binding the Flask development server to public interfaces (`0.0.0.0`) with `debug=True` allows remote code execution.
**Prevention:** Always set `debug=False` when binding to public interfaces, or run behind a reverse proxy in production.
