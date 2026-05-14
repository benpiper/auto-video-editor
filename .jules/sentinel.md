## 2026-05-14 - Path Traversal in Secondary JSON Endpoints
**Vulnerability:** Secondary `/api/jobs` endpoint did not sanitize user-provided filename from JSON payload, leading to an out-of-bounds file read via `../../../`.
**Learning:** Endpoints retrieving files based on secondary handles must independently sanitize inputs. Relying solely on the primary upload logic fails when a spoofed payload passes a malicious path to downstream workflows.
**Prevention:** Always apply `werkzeug.utils.secure_filename` (or similar) consistently to any endpoint constructing file paths from user inputs, including background tasks or secondary endpoints.
