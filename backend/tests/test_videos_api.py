import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import CurrentUser, current_user
from app.core.db import get_db
from app.main import app
from app.services import dedupe, retrieval, storage
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
    with patch.object(storage, "upload_fileobj") as upload, patch.object(tasks.ingest_video, "delay") as delay,          patch.object(dedupe, "find_by_sha256", return_value=None) as find:  # no duplicate unless a test says so
        c = TestClient(app)  # no `with`: skips lifespan, so no storage bucket call
        c.upload, c.delay, c.find = upload, delay, find
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


def test_list_videos_includes_created_at_for_the_admin_table(client, db):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    created = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    db.scalars.return_value.all.return_value = [
        SimpleNamespace(id=VIDEO_ID, title="Talk", status="ready", duration_sec=1700, created_at=created),
        SimpleNamespace(id=uuid.uuid4(), title="New", status="uploaded", duration_sec=None, created_at=created)]
    r = client.get("/api/videos")
    assert r.status_code == 200
    assert r.json()[0] == {"id": str(VIDEO_ID), "title": "Talk", "status": "ready", "duration_sec": 1700,
                           "created_at": "2026-03-04T05:06:07Z"}
    assert r.json()[1]["duration_sec"] is None


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
    db.get.return_value = MagicMock(language="ta-IN")
    rows = [{"idx": 0, "start_sec": 0.0, "end_sec": 30.0, "text": "a", "text_en": "A", "text_roman": "a."},
            {"idx": 1, "start_sec": 25.0, "end_sec": 55.0, "text": "b", "text_en": None, "text_roman": None}]  # 2nd: translation pending
    with patch.object(retrieval, "list_chunks", return_value=rows):
        r = client.get(f"/api/videos/{VIDEO_ID}/transcript")
    assert r.status_code == 200 and r.json() == {"language": "ta-IN", "chunks": rows}


def test_transcript_language_may_be_unknown(client, db):
    app.dependency_overrides[current_user] = lambda: CurrentUser(id=str(uuid.uuid4()), role="user")
    db.get.return_value = MagicMock(language=None)
    with patch.object(retrieval, "list_chunks", return_value=[]):
        assert client.get(f"/api/videos/{VIDEO_ID}/transcript").json() == {"language": None, "chunks": []}


def test_transcript_404(client, db):
    db.get.return_value = None
    assert client.get(f"/api/videos/{VIDEO_ID}/transcript").status_code == 404


# ---------------------------------------------------------------- duplicate uploads (SHA-256)
import hashlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from sqlalchemy.exc import IntegrityError  # noqa: E402

DATA = b"the same bytes " * 10


def existing(status="ready", title="My talk"):
    return SimpleNamespace(id=uuid.uuid4(), title=title, status=status)


def test_upload_stores_the_sha256_of_the_file(client, db):
    assert post(client, data=DATA).status_code == 202
    assert db.add.call_args_list[0][0][0].sha256 == hashlib.sha256(DATA).hexdigest()
    assert client.find.call_args[0][1] == hashlib.sha256(DATA).hexdigest()


def test_the_stored_file_is_the_whole_file_after_hashing(client):
    """Hashing reads the spool to the end: the upload must still start from byte 0."""
    seen = {}
    client.upload.side_effect = lambda f, key, ctype: seen.update(first=f.read(5), tell=f.tell())
    assert post(client, data=DATA).status_code == 202
    assert seen["first"] == DATA[:5]


def test_duplicate_returns_409_with_the_existing_videos_id_and_stores_nothing(client, db):
    found = existing()
    client.find.return_value = found
    r = post(client, data=DATA)
    assert r.status_code == 409
    body = r.json()
    assert body["video_id"] == str(found.id) and body["title"] == "My talk" and body["status"] == "ready"
    assert str(found.id) in body["detail"] and '"My talk"' in body["detail"]
    client.upload.assert_not_called()  # nothing went to S3
    db.add.assert_not_called()
    client.delay.assert_not_called()


@pytest.mark.parametrize("status,hint", [("failed", "Reprocess"), ("uploaded", "still being processed"),
                                         ("ready", "Delete that video first")])
def test_duplicate_message_says_what_to_do_for_each_state(client, status, hint):
    client.find.return_value = existing(status=status)
    assert hint in post(client, data=DATA).json()["detail"]


def test_concurrent_duplicate_is_a_409_and_the_stored_object_is_removed(client, db):
    """Two identical uploads pass the check together; the unique index rejects the second insert."""
    winner = existing()
    client.find.side_effect = [None, winner]  # not there at the check, there after the failed insert
    db.commit.side_effect = IntegrityError("insert", {}, Exception("videos_sha256_key"))
    with patch.object(storage, "delete") as delete:
        r = post(client, data=DATA)
    assert r.status_code == 409 and r.json()["video_id"] == str(winner.id)
    delete.assert_called_once_with(client.upload.call_args[0][1])
    db.rollback.assert_called()
    client.delay.assert_not_called()


def test_integrity_error_that_is_not_a_duplicate_stays_a_500(client, db):
    client.find.side_effect = [None, None]
    db.commit.side_effect = IntegrityError("insert", {}, Exception("something else"))
    with patch.object(storage, "delete"):
        r = post(client, data=DATA)
    assert r.status_code == 500 and "Could not save" in r.json()["detail"]


def test_different_files_are_not_duplicates(client, db):
    assert post(client, data=b"a" * 50).status_code == 202
    assert post(client, data=b"b" * 50).status_code == 202
    first, second = (c[0][1] for c in client.find.call_args_list)
    assert first != second


# ---------------------------------------------------------------- against the real database
def test_real_database_rejects_the_same_file_twice(api, db_session):
    """init.sql has the unique index; the API answers 409 for the second identical upload and keeps one row."""
    from sqlalchemy import text
    api.post("/auth/register", json={"email": "root@example.com", "password": "Passw0rd123"})
    db_session.execute(text("UPDATE users SET role = 'admin' WHERE email = 'root@example.com'"))
    db_session.commit()
    token = api.post("/auth/login", json={"email": "root@example.com", "password": "Passw0rd123"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    with patch.object(storage, "upload_fileobj"), patch.object(tasks.ingest_video, "delay"):
        first = api.post("/api/videos", files={"file": ("a.mp4", DATA, "video/mp4")}, headers=headers, data={"title": "Talk"})
        second = api.post("/api/videos", files={"file": ("renamed.mp4", DATA, "video/mp4")}, headers=headers)
        other = api.post("/api/videos", files={"file": ("b.mp4", DATA + b"!", "video/mp4")}, headers=headers)
    assert first.status_code == 202 and other.status_code == 202
    assert second.status_code == 409  # a different file NAME does not matter, the bytes do
    assert second.json()["video_id"] == first.json()["video_id"]
    rows = db_session.execute(text("SELECT sha256 FROM videos ORDER BY created_at")).scalars().all()
    assert len(rows) == 2 and rows[0] == hashlib.sha256(DATA).hexdigest()


def test_unique_index_allows_many_videos_without_a_hash(api, db_session):
    from sqlalchemy import text
    for i in range(3):
        db_session.execute(text("INSERT INTO videos (id, title, storage_key) VALUES (:i, 't', 'raw/x.mp4')"), {"i": uuid.uuid4()})
    db_session.commit()  # three NULL hashes: legacy rows must coexist
    assert db_session.execute(text("SELECT count(*) FROM videos WHERE sha256 IS NULL")).scalar() == 3


# ---------------------------------------------------------------- language chosen on upload
def test_the_chosen_language_is_stored_as_the_videos_hint(client, db):
    assert post(client, language="ta-IN").status_code == 202
    assert db.add.call_args_list[0][0][0].language_hint == "ta-IN"


@pytest.mark.parametrize("value,stored", [("", None), ("auto", None), ("hi-in", "hi-IN"), (" en-IN ", "en-IN")])
def test_no_choice_or_a_sloppy_spelling_is_normalised(client, db, value, stored):
    assert post(client, language=value).status_code == 202
    assert db.add.call_args_list[0][0][0].language_hint == stored


def test_leaving_the_language_out_means_automatic(client, db):
    assert post(client).status_code == 202
    assert db.add.call_args_list[0][0][0].language_hint is None


@pytest.mark.parametrize("bad", ["klingon", "hi", "en-US", "x" * 21])
def test_an_unknown_language_is_refused_before_anything_is_stored(client, db, bad):
    r = post(client, language=bad)
    assert r.status_code == 422
    client.upload.assert_not_called()
    db.add.assert_not_called()
    client.delay.assert_not_called()
