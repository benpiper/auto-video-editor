import os
import pytest
from core.db import Project, Task
from core.cleanup import cleanup_orphaned_files
from web_app.app import app as flask_app
from peewee import SqliteDatabase

test_db = SqliteDatabase(":memory:")


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    models = [Project, Task]
    test_db.bind(models, bind_refs=False, bind_backrefs=False)
    test_db.connect()
    test_db.create_tables(models)

    with flask_app.test_client() as client:
        yield client

    test_db.drop_tables(models)
    test_db.close()


def test_project_deletion_removes_files(client, tmp_path):
    """Test that DELETE /api/projects/<id> removes files from disk."""
    # Create dummy files
    in_file = tmp_path / "input.mp4"
    out_file = tmp_path / "output.mp4"
    in_file.write_text("fake video")
    out_file.write_text("fake edited video")

    project = Project.create(
        name="Cleanup Test", input_path=str(in_file), output_path=str(out_file)
    )

    response = client.delete(f"/api/projects/{project.id}")
    assert response.status_code == 200

    # Check disk
    assert not os.path.exists(in_file)
    assert not os.path.exists(out_file)

    # Check DB
    assert Project.select().where(Project.id == project.id).count() == 0


def test_orphan_cleanup(tmp_path):
    """Test that cleanup_orphaned_files removes unreferenced files."""
    # Setup test DB tables for this standalone test
    models = [Project, Task]
    test_db.bind(models, bind_refs=False, bind_backrefs=False)
    test_db.connect()
    test_db.create_tables(models)

    try:
        # Setup directories
        uploads = tmp_path / "uploads"
        outputs = tmp_path / "outputs"
        uploads.mkdir()
        outputs.mkdir()

        # Referenced file
        ref_file = uploads / "kept.mp4"
        ref_file.write_text("stay")

        # Orphaned files
        orphan_upload = uploads / "gone_in.mp4"
        orphan_output = outputs / "gone_out.mp4"
        orphan_upload.write_text("delete")
        orphan_output.write_text("delete")

        # Mock DB
        Project.create(name="Keep", input_path=str(ref_file.absolute()))

        # Run cleanup
        cleanup_orphaned_files(str(uploads), str(outputs))

        assert os.path.exists(ref_file)
        assert not os.path.exists(orphan_upload)
        assert not os.path.exists(orphan_output)
    finally:
        test_db.drop_tables(models)
        test_db.close()
