
## 2024-05-24 - Werkzeug Debugger Exposure
**Vulnerability:** Flask application was configured with `debug=True` while binding to `host='0.0.0.0'`, exposing the Werkzeug interactive debugger to all network interfaces.
**Learning:** Developers often leave debug mode enabled for local convenience, but combining it with a public bind address creates a critical Remote Code Execution (RCE) vulnerability.
**Prevention:** Always enforce `debug=False` when binding to `0.0.0.0` or public IPs, and manage debug mode strictly through environment variables.
