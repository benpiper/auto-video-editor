## 2024-05-24 - Disabled Werkzeug Debugger on Public Interface
**Vulnerability:** Flask application bound to 0.0.0.0 with debug=True enabled.
**Learning:** Exposing the Werkzeug interactive debugger on public interfaces allows remote unauthenticated attackers to execute arbitrary Python code on the server.
**Prevention:** Always ensure debug=False is set in production or when binding to public host IPs like 0.0.0.0.
