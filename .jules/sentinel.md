## 2024-05-15 - Secondary Endpoint Input Sanitization Bypass
**Vulnerability:** Path traversal in `/api/jobs` via unsanitized JSON inputs (`filename`, `original_filename`, `bg_image`).
**Learning:** Secondary endpoints that receive filenames via client JSON requests must independently sanitize those inputs using `secure_filename`. Relying only on primary upload endpoint sanitization allows attackers to bypass restrictions via spoofed JSON payloads.
**Prevention:** Always apply `secure_filename` to any user-supplied string used to construct a file path, regardless of whether the file was supposedly uploaded in a previous step.
