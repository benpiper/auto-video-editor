## 2024-06-19 - Werkzeug Debugger RCE Exposure
**Vulnerability:** Flask `debug=True` was active while binding to `host='0.0.0.0'`, exposing the Werkzeug interactive debugger to all network interfaces.
**Learning:** Developers often leave debug mode enabled for easier troubleshooting, but binding to all interfaces makes the RCE-capable debugger accessible externally.
**Prevention:** Always ensure `debug=False` in production or when binding to public interfaces (`0.0.0.0`). Use environment variables to control debug mode explicitly.
