# Auto Video Editor Documentation Index

Welcome to the internal documentation for the Auto Video Editor project. This collection serves as the primary source of truth for development, architecture, and system integration.

## 🌟 Project Overview
Auto Video Editor is an AI-enhanced toolkit designed to automate the boring parts of video editing. It focuses on silence removal, filler word elimination, and background subtraction.

- **Primary Language**: Python
- **Architecture Style**: Modular Pipeline / Multi-part Monolith
- **UI Framework**: Flask + Vanilla JS

## 🗺️ Part-Based Navigation

### Core Processing Hub (`core`)
- **Type**: Backend Logic / CLI
- **Tech Stack**: Whisper, MoviePy, FFmpeg, Torch
- **[Core Architecture](./architecture-core.md)**
- **[CLI Reference](./api-contracts.md#2-command-line-interface-core)**

### Web Dashboard & API (`web_portal`)
- **Type**: Web Application
- **Tech Stack**: Flask, SSE, Swagger
- **[Web Architecture](./architecture-web_portal.md)**
- **[REST API Docs](./api-contracts.md#1-rest-api-web_portal)**
- **[UI Inventory](./ui-component-inventory.md)**

## 📚 Complete Documentation

### Planning & Architecture
- [Project Overview](./project-overview.md)
- [System Architecture - Core](./architecture-core.md)
- [System Architecture - Web Portal](./architecture-web_portal.md)
- [Integration Architecture](./integration-architecture.md)

### Technical Specifications
- [API Contracts](./api-contracts.md)
- [Data Models](./data-models.md)
- [Source Tree Analysis](./source-tree-analysis.md)

### Guides
- [Development Guide](./development-guide.md)
- [Logging Guide](../LOGGING_GUIDE.md)
- [Quality Setting Guide](../QUALITY_GUIDE.md)
- [UV Usage Guide](../UV_USAGE.md)

## 🛠️ Getting Started

1.  **System FFmpeg**: Ensure `ffmpeg` is installed on your system.
2.  **Environment Setup**: Run `uv sync` to install all dependencies.
3.  **Core Test**: Run `uv run python create_test_video.py` to generate sample data.
4.  **Launch Web**: Run `uv run python web_app/app.py` and visit `http://localhost:5000`.
