## 2024-05-19 - Remote Code Execution via exposed Werkzeug debugger
**Vulnerability:** The Flask application was binding to a public interface (`host='0.0.0.0'`) while `debug=True` was set in `app.run()`. This configuration exposes the Werkzeug interactive debugger to the entire network, leading to unauthenticated arbitrary remote code execution (RCE).
**Learning:** Running an application in debug mode is extremely dangerous when bound to all network interfaces, as debugging tools are explicitly designed to allow code execution and bypass standard security controls.
**Prevention:** Always explicitly set `debug=False` in production code or when binding to public network interfaces (`0.0.0.0`), restricting debug mode exclusively to `localhost` (`127.0.0.1`) during local development.
