# Story 2.3: Project Management Dashboard

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As an editor,
I want a visual list of my video projects and their current status,
so that I can manage my editing workload efficiently.

## Acceptance Criteria

1. **Given** multiple projects in the database, **When** I visit the `/dashboard` route, **Then** the UI must list all projects in a clean, modern grid or list view.
2. **Given** a list of projects, **When** displayed, **Then** each project must show its name, input path, and a clear status badge (e.g., Ingested, Detecting, Ready, Complete, Error).
3. **Given** the dashboard view, **When** projects are loaded, **Then** the data must be fetched from the `api/projects` endpoint.
4. **Given** a project in the list, **When** clicked, **Then** it should provide a way to navigate to the detailed editor view (to be implemented in later stories).

## Tasks / Subtasks

- [x] Implement `GET /api/projects` endpoint in `web_app/api.py` (AC: 1, 3)
  - [x] Fetch all projects from the database
  - [x] Return a list of project objects (serialized)
- [x] Create `web_app/templates/dashboard.html` (AC: 1, 2)
  - [x] Implement a premium, modern design using CSS
  - [x] Use fetch API to pull projects from `/api/projects`
  - [x] Render project status badges dynamically
- [x] Implement `/dashboard` route in `web_app/app.py` (AC: 1)
  - [x] Render the `dashboard.html` template
- [x] Implement unit/integration tests (AC: 1, 3)
  - [x] Test API endpoint returns project list
  - [x] Test UI rendering with mock data

## Dev Notes

- **Aesthetics**: Implemented a modern glassmorphism grid with dynamic status badges.
- **Resilience**: Fixed an over-aggressive `.gitignore` rule (`temp*`) that was blocking the `templates/` directory.
- **Performance**: Used efficient Peewee queries with descending order for the latest projects first.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 2.3]

## Dev Agent Record

### Agent Model Used

Antigravity (Current Turn)

### Debug Log References

- Pytest run: `tests/test_api.py` passed 100% (3/3).

### Completion Notes List

- Implemented `GET /api/projects` API for data retrieval.
- Created `dashboard.html` with a modern CSS grid layout.
- Fixed `.gitignore` bug that was impacting template development.
- Verified dashboard route and project listing with unit and integration tests.

### File List
- `web_app/api.py`
- `web_app/app.py`
- `web_app/templates/dashboard.html`
- `web_app/static/css/dashboard.css`
- `tests/test_api.py`
