import pytest
from peewee import SqliteDatabase
from core.db import Project, Task

# Use a test database
test_db = SqliteDatabase(":memory:")


@pytest.fixture(autouse=True)
def setup_test_db():
    """Sets up an in-memory database for each test."""
    # Bind models to test_db
    models = [Project, Task]
    test_db.bind(models, bind_refs=False, bind_backrefs=False)
    test_db.connect()
    test_db.create_tables(models)
    yield
    test_db.drop_tables(models)
    test_db.close()


def test_project_creation():
    """Test creating a project and verifying defaults."""
    project = Project.create(name="Test Project", input_path="/tmp/test.mp4")

    assert project.id is not None
    assert project.status == "ingested"
    assert project.name == "Test Project"
    assert isinstance(project.metadata, dict)


def test_task_creation():
    """Test creating a task associated with a project."""
    project = Project.create(name="Task Project", input_path="inv.mp4")
    task = Task.create(project=project, task_type="ingest", status="running")

    assert task.project.id == project.id
    assert task.status == "running"
    assert task.progress == 0


def test_json_field():
    """Test the custom JSONField."""
    meta = {"width": 1920, "height": 1080, "tags": ["4k", "clean"]}
    project = Project.create(name="JSON Project", input_path="p.mp4", metadata=meta)

    # Reload from DB
    retrieved = Project.get_by_id(project.id)
    assert retrieved.metadata == meta
    assert retrieved.metadata["width"] == 1920


def test_cascade_delete():
    """Verify that deleting a project also deletes its tasks."""
    project = Project.create(name="Delete Me", input_path="d.mp4")
    Task.create(project=project, task_type="test")

    assert Task.select().where(Task.project == project).count() == 1

    project.delete_instance(recursive=True)

    assert Task.select().where(Task.project == project).count() == 0
