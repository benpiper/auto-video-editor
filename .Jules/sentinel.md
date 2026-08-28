## 2026-08-28 - Prevent Werkzeug Debugger Exposure
**Vulnerability:** Flask development server started with debug mode enabled (debug=True) and bound to a public interface (0.0.0.0), exposing the Werkzeug interactive debugger and enabling Remote Code Execution (RCE).
**Learning:** Development configurations left active in production-like entry points open severe RCE vulnerabilities.
**Prevention:** Explicitly disable debug mode or manage it via environment variables when binding Flask to public interfaces.
