import os
import uuid
import json
from datetime import datetime
from peewee import (
    SqliteDatabase,
    Model,
    CharField,
    TextField,
    DateTimeField,
    ForeignKeyField,
    IntegerField,
    FloatField,
    BooleanField,
)

# Database location
DB_PATH = os.getenv("DB_PATH", "projects.db")
db = SqliteDatabase(DB_PATH)


class JSONField(TextField):
    """Simple JSON field for Peewee."""

    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        if value is not None:
            return json.loads(value)
        return {}


class BaseModel(Model):
    class Meta:
        database = db


class Project(BaseModel):
    """Represents a video editing project."""

    id = CharField(primary_key=True, default=lambda: str(uuid.uuid4()))
    name = CharField()
    input_path = CharField()
    output_path = CharField(null=True)
    status = CharField(
        default="ingested"
    )  # ingested, detecting, processing, complete, error
    created_at = DateTimeField(default=datetime.now)
    metadata = JSONField(default={})


class Task(BaseModel):
    """Represents a background job/task associated with a project."""

    id = CharField(primary_key=True, default=lambda: str(uuid.uuid4()))
    project = ForeignKeyField(Project, backref="tasks", on_delete="CASCADE")
    task_type = CharField()  # ingest, detect_silence, remove_filler, render
    status = CharField(default="pending")  # pending, running, complete, error
    progress = IntegerField(default=0)
    error_message = TextField(null=True)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        return super(Task, self).save(*args, **kwargs)


class CutCandidate(BaseModel):
    """Represents a proposed cut (silence or filler word)."""

    id = CharField(primary_key=True, default=lambda: str(uuid.uuid4()))
    project = ForeignKeyField(Project, backref="cuts", on_delete="CASCADE")
    type = CharField()  # silence, filler
    start_time = FloatField()
    end_time = FloatField()
    ignored = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.now)


def init_db():
    """Initializes the database and creates tables."""
    db.connect()
    db.create_tables([Project, Task, CutCandidate], safe=True)
    db.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
