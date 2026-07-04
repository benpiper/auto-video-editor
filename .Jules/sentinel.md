## 2024-07-04 - Werkzeug Debugger Exposed on Public Interface
**Vulnerability:** The Flask application was bound to a public interface (0.0.0.0) with debug=True enabled.
**Learning:** Running Flask with debug mode on a public interface exposes the Werkzeug interactive debugger, which can allow remote code execution (RCE).
**Prevention:** Always ensure debug=False is set when binding to a publicly accessible IP address.
