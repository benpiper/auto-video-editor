## 2024-05-18 - Werkzeug Interactive Debugger Exposure
**Vulnerability:** Flask application bound to '0.0.0.0' with debug=True exposed the Werkzeug interactive debugger to public interfaces.
**Learning:** The Werkzeug interactive debugger allows execution of arbitrary Python code, leading to severe Remote Code Execution (RCE) vulnerabilities if exposed outside of localhost.
**Prevention:** Always ensure debug=False is set when binding to public network interfaces ('0.0.0.0') to prevent debugger exposure.
