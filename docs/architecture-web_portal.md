# Architecture: Web Portal

The Web Portal provides a user-friendly interface and automation API for the video processor.

## 1. Executive Summary
A Flask-based web application that manages video uploads, job queuing (threading), and real-time status reporting. It exposes both a GUI and a documented REST API.

## 2. Technology Stack
- **Languages**: Python, Vanilla JavaScript, CSS3, HTML5
- **Framework**: Flask
- **Integration**: SSE (Server-Sent Events) for live updates
- **Documentation**: Swagger UI / OpenAPI

## 3. Architecture Pattern
**MVC (Model-View-Controller)**:
- **Models**: `Job` class in `state.py` (In-memory).
- **Views**: `index.html` template and AJAX handlers in `app.js`.
- **Controllers**: `app.py` and `api.py` route handlers.

## 4. Key Components
- `app.py`: Main router and file system manager.
- `state.py`: The global state store for active jobs.
- `api.py`: Clean REST interface.
- `app.js`: Client-side logic for file manipulation and event listening.

## 5. Deployment Architecture
Designed to run as a single instance.
- **Concurrency**: Handled via Python threads (one thread per processing job).
- **Storage**: Direct filesystem access to `uploads/` and `outputs/`.
