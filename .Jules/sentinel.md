## 2024-06-26 - Exposed Werkzeug Debugger on Public Interface
**Vulnerability:** The Flask application was binding to `0.0.0.0` with `debug=True`, exposing the Werkzeug interactive debugger to all network interfaces, which allows for Remote Code Execution (RCE).
**Learning:** Hardcoding `debug=True` when binding to a public interface is a dangerous default that accidentally shipped, bypassing the expected security perimeter.
**Prevention:** Always set `debug=False` when binding to `host='0.0.0.0'` in a production-like or shared network environment to prevent exposing the interactive debugger.
