## 2024-07-05 - Fix Werkzeug Interactive Debugger Exposure
**Vulnerability:** The Flask application was running with `debug=True` while binding to all public interfaces (`host='0.0.0.0'`).
**Learning:** Exposing the Werkzeug interactive debugger to the public internet creates a critical Remote Code Execution (RCE) risk, allowing attackers to execute arbitrary Python code on the server.
**Prevention:** Always ensure `debug=False` is explicitly set when binding Flask applications to public interfaces like `0.0.0.0`.
