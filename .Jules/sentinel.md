## 2026-09-01 - Werkzeug Interactive Debugger Exposure
**Vulnerability:** Flask debug mode was enabled on a public interface (0.0.0.0), exposing the Werkzeug interactive debugger.
**Learning:** Binding to 0.0.0.0 with debug=True allows arbitrary remote code execution. Development configurations must be disabled when exposed.
**Prevention:** Always verify debug=False when host is set to 0.0.0.0 in Flask entry points.
