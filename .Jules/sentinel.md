## 2024-06-30 - Werkzeug Debugger Exposure
**Vulnerability:** Flask development server was configured with `debug=True` while bound to `host='0.0.0.0'`.
**Learning:** Exposing the Werkzeug interactive debugger on public network interfaces allows unauthenticated remote code execution (RCE).
**Prevention:** Always ensure `debug=False` in production-like bindings or when `host='0.0.0.0'` is used, and rely on secure logging instead.
