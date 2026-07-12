## 2026-07-12 - Flask Debug Mode RCE Exposure
**Vulnerability:** Werkzeug interactive debugger enabled on a public interface (`app.run(debug=True, host='0.0.0.0')`).
**Learning:** The Werkzeug interactive debugger permits remote code execution. Exposing it on `0.0.0.0` allows any user who can reach the server to execute arbitrary Python code.
**Prevention:** Always ensure `debug=False` when binding to public interfaces like `0.0.0.0` in production-like configurations.
