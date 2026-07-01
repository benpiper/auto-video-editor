## 2024-05-20 - Flask Werkzeug Debugger RCE Risk
**Vulnerability:** Flask development server was running with `debug=True` while bound to `host='0.0.0.0'`.
**Learning:** This exposes the Werkzeug interactive debugger to all network interfaces, allowing remote code execution.
**Prevention:** Always set `debug=False` when binding to `0.0.0.0` or public interfaces. Use environment variables to control debug mode.
