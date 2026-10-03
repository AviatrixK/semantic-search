import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import CurrentUser, current_user
from app.core.db import get_db
from app.main import app
from app.services import retrieval, storage
from app.workers import tasks

MB = 1024 * 1024
VIDEO_ID = uuid.uuid4()


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def client(db):
    """Admin by default. The real require_admin runs on top of the overridden current_user."""
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[current_user] = lambda: CurrentUser(id=str(uuid.uuid4()), role="admin")
    with patch.object(storage, "upload_fileobj") as upload, patch.object(tasks.ingest_video, "delay") as delay:
        c = TestClient(app)  # no `with`: skips lifespan, so no storage bucket call
        c.upload, c.delay = upload, delay
        yield c
    app.dependency_overrides.clear()


def post(client, name="clip.mp4", ctype="video/mp4", data=b"x" * 100, **form):
    return client.post("/api/videos", files={"file": (name, data, ctype)}, data=form)


def test_upload_ok_defaults_title_to_filename(client, db):
    r = post(client)
    assert r.status_code == 202
    assert db.add.call_args_list[0][0][0].title == "clip.mp4"
    key = client.upload.call_args[0][1]
    assert key.startswith("raw/") and key.endswith(".mp4")
    client.delay.assert_called_once()


def test_upload_inserts_video_before_job(client, db):
    order = []
    db.add.side_effect = lambda obj: order.append(type(obj).__name__)
    db.flush.side_effect = lambda: order.append("flush")
    assert post(client).status_code == 202
    assert order == ["Video", "flush", "Job"]  # job FK needs the video row to exist first


def test_upload_db_failure_removes_stored_file_and_returns_readable_error(client, db):
    db.commit.side_effect = SQLAlchemyError("connection lost")
    with patch.object(storage, "delete") as delete:
        r = post(client)
    assert r.status_code == 500 and r.json() == {"detail": "Could not save the video. Please try again."}
    uploaded_key = client.upload.call_args[0][1]
    delete.assert_called_once_with(uploaded_key)
    db.rollback.assert_called()
    client.delay.assert_not_called()


def test_upload_db_failure_still_readable_if_cleanup_fails(client, db):
    db.flush.side_effect = SQLAlchemyError("fk violation")
    with patch.object(storage, "delete", side_effect=RuntimeError("storage down")):
        r = post(client)
    assert r.status_code == 500 and "Could not save" in r.json()["detail"]


def test_upload_uses_title_field(client, db):
    assert post(client, title="  My talk  ").status_code == 202
    assert db.add.call_args_list[0][0][0].title == "My talk"


@pytest.mark.parametrize("name,ctype", [("notes.txt", "text/plain"), ("clip.exe", "video/mp4"),
                                        ("noextension", "video/mp4"), ("clip.mp4", "text/plain")])
def test_upload_rejects_bad_type_or_extension(client, name, ctype):
    assert post(client, name, ctype).status_code == 415
    client.upload.assert_not_called()
    client.delay.assert_not_called()


def test_upload_rejects_empty_and_oversized(client):
    assert post(client, data=b"").status_code == 400
    assert post(client, data=b"x" * (MB + 1)).status_code == 413  # conftest sets MAX_UPLOAD_MB=1
    client.upload.assert_not_called()


def test_upload_requires_admin(client):
    app.dependency_overrides[current_user] = lambda: CurrentUser(id=str(uuid.uuid4()), role="user")
    assert post(client).status_code == 403


def test_endpoints_require_login():
    app.dependency_overrides.clear()
    c = TestClient(app)
    for method, path in [("post", "/api/videos"), ("post", f"/api/videos/{VIDEO_ID}/reprocess"),
                         ("get", f"/api/videos/{VIDEO_ID}/transcript")]:
        assert getattr(c, method)(path).status_code == 401, path


def test_reprocess_enqueues_new_job(client, db):
    db.get.return_value = MagicMock(status="ready")
    db.scalar.return_value = None  # no active job
    r = client.post(f"/api/videos/{VIDEO_ID}/reprocess")
    assert r.status_code == 202 and r.json()["video_id"] == str(VIDEO_ID)
    assert db.get.return_value.status == "uploaded"
    client.delay.assert_called_once()


def test_reprocess_404_and_409(client, db):
    db.get.return_value = None
    assert client.post(f"/api/videos/{VIDEO_ID}/reprocess").status_code == 404
    db.get.return_value = MagicMock()
    db.scalar.return_value = uuid.uuid4()  # a job is still running
    assert client.post(f"/api/videos/{VIDEO_ID}/reprocess").status_code == 409
    client.delay.assert_not_called()


def test_reprocess_requires_admin(client):
    app.dependency_overrides[current_user] = lambda: CurrentUser(id=str(uuid.uuid4()), role="user")
    assert client.post(f"/api/videos/{VIDEO_ID}/reprocess").status_code == 403


def test_transcript_returns_ordered_chunks(client, db):
    app.dependency_overrides[current_user] = lambda: CurrentUser(id=str(uuid.uuid4()), role="user")  # any user
    db.get.return_value = MagicMock()
    rows = [{"idx": 0, "start_sec": 0.0, "end_sec": 30.0, "text": "a"},
            {"idx": 1, "start_sec": 25.0, "end_sec": 55.0, "text": "b"}]
    with patch.object(retrieval, "list_chunks", return_value=rows):
        r = client.get(f"/api/videos/{VIDEO_ID}/transcript")
    assert r.status_code == 200 and r.json() == rows


def test_transcript_404(client, db):
    db.get.return_value = None
    assert client.get(f"/api/videos/{VIDEO_ID}/transcript").status_code == 404
