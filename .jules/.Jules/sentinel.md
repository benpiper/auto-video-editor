## $(date +%Y-%m-%d) - Prevent Flask RCE and 500 Errors
**Vulnerability:** Werkzeug interactive debugger exposed on 0.0.0.0 via `debug=True`, allowing RCE. Missing input validation on Flask route type conversions causing 500 Internal Server Errors.
**Learning:** Development configurations often leak into production-like binding interfaces, and form data must always be treated as untrusted strings requiring safe casting.
**Prevention:** Always set `debug=False` when binding to 0.0.0.0. Wrap all `request.form.get` type casts in `try...except ValueError` blocks.
