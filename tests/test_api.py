import pytest
import json
from web_app.app import app as flask_app
from core.db import Project, Task, db as main_db
from peewee import SqliteDatabase

# Use an in-memory database for testing
test_db = SqliteDatabase(":memory:")


@pytest.fixture
def client():
    """Flask test client fixture."""
    flask_app.config["TESTING"] = True

    # Bind models to test_db
    models = [Project, Task]
    test_db.bind(models, bind_refs=False, bind_backrefs=False)
    test_db.connect()
    test_db.create_tables(models)

    with flask_app.test_client() as client:
        yield client

    test_db.drop_tables(models)
    test_db.close()


def test_list_projects_empty(client):
    """Test getting projects when none exist."""
    response = client.get("/api/projects")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) == 0


def test_list_projects_with_data(client):
    """Test getting projects when they exist in the database."""
    Project.create(name="Project 1", input_path="/path/1.mp4", status="complete")
    Project.create(name="Project 2", input_path="/path/2.mp4", status="processing")

    response = client.get("/api/projects")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 2
    assert data[0]["name"] == "Project 2"  # Descending order
    assert data[1]["name"] == "Project 1"
    assert data[0]["status"] == "processing"


def test_dashboard_route(client):
    """Test that the dashboard route renders."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Dashboard" in response.data
    assert b"project-list" in response.data
