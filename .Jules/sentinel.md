## 2026-07-11 - Exposed Werkzeug Debugger
**Vulnerability:** Flask application bound to 0.0.0.0 with debug=True, exposing the Werkzeug interactive debugger to all network interfaces.
**Learning:** In Flask, enabling debug mode while binding to public interfaces allows remote arbitrary code execution via the debugger console.
**Prevention:** Always ensure debug=False is explicitly set when binding to 0.0.0.0 in application runner scripts.
