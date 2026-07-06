## 2024-05-24 - Werkzeug Debugger RCE Risk on Public Interfaces
**Vulnerability:** The Flask application was bound to all interfaces (`host='0.0.0.0'`) while `debug=True` was enabled in `app.run()`.
**Learning:** Binding a Flask app with debug mode enabled to public interfaces exposes the Werkzeug interactive debugger, which can be exploited for Remote Code Execution (RCE) by unauthorized users.
**Prevention:** Always ensure `debug=False` is set when binding to `0.0.0.0` or public network interfaces. Use local environment variables or secure configuration management to toggle debug mode only for local development on `127.0.0.1`.
