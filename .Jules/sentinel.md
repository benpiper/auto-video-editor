## 2026-08-24 - Werkzeug Interactive Debugger Exposure
**Vulnerability:** Flask application was running with `debug=True` while bound to a public interface (`0.0.0.0`).
**Learning:** Exposing the Werkzeug interactive debugger on a public interface allows an attacker to execute arbitrary Python code remotely.
**Prevention:** Ensure development configurations like `debug=True` are not active when binding to public interfaces (`0.0.0.0`).
