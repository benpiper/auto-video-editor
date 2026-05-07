## 2024-05-07 - Secondary Endpoint Path Traversal
**Vulnerability:** The `/api/jobs` endpoint read file paths (`filename`, `bg_image`) directly from JSON payloads without sanitization, allowing arbitrary file read/deletion via path traversal (e.g., `../../etc/passwd`).
**Learning:** Path traversal isn't just an upload issue. Any endpoint acting on filenames from client payloads must independently sanitize inputs.
**Prevention:** Always use `werkzeug.utils.secure_filename` to sanitize any user-supplied filename or file path in all endpoints, not just primary upload routes.
