# Story 2.5: Project Cleanup & Data Persistence Logic

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a user,
I want the system to automatically purge temporary render files,
so that my local storage doesn't get cluttered.

## Acceptance Criteria

1. **Given** a background video processing task is complete, **When** the final output file is successfully generated, **Then** all intermediate artifacts (e.g., extracted audio wavs, temporary video chunks) must be deleted from the temporary storage (RAMDisk or uploads).
2. **Given** a request to delete a project, **When** the project is removed from the database, **Then** its associated input/output files on disk must also be deleted (or archived if configured).
3. **Given** the application's lifecycle, **When** it starts up, **Then** it should optionally perform a sanity check to clean up orphaned temporary files from previous crashed runs.

## Tasks / Subtasks

- [x] Implement Project Deletion API in `web_app/api.py` (AC: 2)
  - [x] Create `DELETE /api/projects/<id>` endpoint
  - [x] Delete files from disk before removing database record
- [x] Enhance `processor.py` (or caller) with better cleanup logic (AC: 1)
  - [x] Ensure `finally` blocks handle intermediate file removal (wavs, chunks, transcripts)
- [x] Implement a basic "Orphan Cleanup" utility in `core/cleanup.py` (AC: 3)
  - [x] Scan `uploads` and `outputs` for files not referenced in the database
- [x] Implement unit tests for cleanup (AC: 1, 2, 3)
  - [x] Test that project deletion actually removes files from disk
  - [x] Test cleanup of temporary files after a mock processing run

## Dev Notes

- **Robustness**: Integrated `os.path.abspath` comparison in the cleanup utility to prevent accidental deletion due to relative path mismatches.
- **Resilience**: The `finally` block in `processor.py` now specifically targets `.wav`, `_transcript.txt`, and transposed temporary videos.
- **API**: The DELETE endpoint returns 200 even if some files were already missing, ensuring the DB record is always eventually removed.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 2.5]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Pytest run: `tests/test_cleanup.py` passed 100% (2/2).

### Completion Notes List

- Implemented `DELETE /api/projects/<id>` with physical file cleanup.
- Refined `processor.py` lifecycle to purge all transient files (wav, chunks, transcripts).
- Developed `core/cleanup.py` for orphan file detection and removal.
- Passed all cleanup-related unit tests.

### File List
- `core/db.py`
- `web_app/api.py`
- `core/cleanup.py`
- `processor.py`
- `tests/test_cleanup.py`
