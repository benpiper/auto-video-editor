
## 2026-08-25 - Werkzeug Interactive Debugger Exposure
**Vulnerability:** Flask development server was configured with `debug=True` while bound to a public interface (`0.0.0.0`), exposing the Werkzeug interactive debugger which allows unauthenticated Remote Code Execution (RCE).
**Learning:** Development configurations should never be hardcoded and exposed on public interfaces, especially in network-facing scripts like `web_app/run.py`.
**Prevention:** Explicitly set `debug=False` in production application entry points, or use environment variables to control debug mode instead of hardcoding `True`.
