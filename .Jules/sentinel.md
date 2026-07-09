## 2026-07-09 - Werkzeug Debugger Exposed
**Vulnerability:** Flask application started with `debug=True` bound to `host='0.0.0.0'`.
**Learning:** Using `debug=True` on public interfaces exposes the Werkzeug interactive debugger, leading to remote code execution.
**Prevention:** Always set `debug=False` when binding to public network interfaces in production or accessible environments.
