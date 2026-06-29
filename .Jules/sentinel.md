## 2024-05-24 - Exposed Werkzeug Debugger
**Vulnerability:** The Flask application was configured to run with `debug=True` while binding to all public interfaces (`host='0.0.0.0'`).
**Learning:** Hardcoding debug mode on public interfaces exposes the interactive Werkzeug debugger, which allows unauthenticated Remote Code Execution (RCE) on the host machine.
**Prevention:** Strictly enforce `debug=False` for any production or externally-facing bindings. Use environment variables to toggle debug mode only for local development (`127.0.0.1`).
