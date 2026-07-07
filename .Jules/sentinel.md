## 2024-07-07 - Exposed Werkzeug Debugger
**Vulnerability:** Flask application was running with `debug=True` while binding to all public interfaces (`host='0.0.0.0'`).
**Learning:** This exposes the Werkzeug interactive debugger, which can enable arbitrary remote code execution (RCE) by anyone who can reach the port.
**Prevention:** Always ensure `debug=False` is set when running Flask on public interfaces.
