# Story 2.1: Local Project Database (SQLite/Peewee)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want a lightweight metadata store for project settings and job states,
so that project data persists across restarts.

## Acceptance Criteria

1. **Given** the application starts for the first time, **When** the database initialization logic runs in `core/db.py`, **Then** it must create a local SQLite database file `projects.db`.
2. **Given** the requirement for structured data, **When** using `Peewee` (or similar ORM), **Then** it must define `Project` and `Task` models with fields for title, input_path, output_path, resolution, and current_status.
3. **Given** a new project, **When** saved to the database, **Then** it must generate a unique UUID and default status of 'ingested'.

## Tasks / Subtasks

- [x] Implement `core/db.py` (AC: 1, 2)
  - [x] Initialize SQLite database connection
  - [x] Define `Project` model (id, name, input_path, status, metadata)
  - [x] Define `Task` model (id, project_id, type, status, progress, error)
  - [x] Implement `init_db()` helper to create tables
- [x] Implement Basic Database CRUD Helpers (AC: 2, 3)
  - [x] Create function to save new projects
  - [x] Create function to update task status/progress
- [x] Update `pyproject.toml` with `peewee` dependency
- [x] Implement unit tests for `core/db.py`
  - [x] Test table creation
  - [x] Test project insertion and retrieval

## Dev Notes

- **Architecture Compliance**:
  - SQLite used for lightweight local persistence.
  - Peewee ORM for clean model definitions.
- **Data Integrity**:
  - Implemented `JSONField` for flexible metadata storage in SQLite.
  - Added recursive deletion for tasks when a project is removed.

### Project Structure Notes

- Database core resides in `core/db.py`.

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Data Management]
- [Source: _bmad-output/planning-artifacts/epics.md#Story 2.1]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Pytest run: `tests/test_db.py` passed 100% (4/4).

### Completion Notes List

- Defined `Project` and `Task` models in `core/db.py`.
- Implemented `init_db` for safe table creation.
- Added `peewee` dependency to `pyproject.toml`.
- Verified model logic, JSON serialization, and cascade deletes with unit tests.

### File List
- `core/db.py`
- `tests/test_db.py`
- `pyproject.toml`
