## 2026-09-03 - Werkzeug Debugger Exposure
**Vulnerability:** Flask development server started with `debug=True` while bound to `0.0.0.0` (all interfaces) in `web_app/run.py` and `web_app/app.py`.
**Learning:** This exposes the Werkzeug interactive debugger, leading to a Remote Code Execution (RCE) vulnerability when the application is exposed externally.
**Prevention:** Never use `debug=True` in production or when binding to a public interface (`0.0.0.0`). Set `debug=False` instead.
