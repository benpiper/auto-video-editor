## 2024-05-24 - Werkzeug Interactive Debugger Exposure
**Vulnerability:** Flask application bound to 0.0.0.0 with debug=True, exposing the Werkzeug interactive debugger.
**Learning:** Leaving debug=True while binding to a public interface allows arbitrary remote code execution via the interactive debugger console.
**Prevention:** Always set debug=False in production or when binding to 0.0.0.0.
