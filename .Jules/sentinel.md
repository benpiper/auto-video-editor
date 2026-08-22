## 2026-08-22 - Flask Debug Mode on Public Interface
**Vulnerability:** Werkzeug interactive debugger exposed to all network interfaces (0.0.0.0) due to `debug=True`, leading to potential remote code execution.
**Learning:** Using `debug=True` with `host='0.0.0.0'` makes the application vulnerable when exposed to the network.
**Prevention:** Always ensure `debug=False` when binding to public interfaces, or strictly bind to `127.0.0.1` when debugging.
