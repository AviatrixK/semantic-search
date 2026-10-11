"""Live job feed: publishing (worker/API side), the snapshot, and GET /api/jobs/events (admin only, SSE)."""
import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import redis
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

from app.api.deps import CurrentUser, current_user
from app.api.routes import jobs
from app.main import app
from app.models import Job, Video
from app.services import jobevents


# ---------------------------------------------------------------- publishing
def test_payload_shape_is_the_job_status_shape():
    jid, vid = uuid.uuid4(), uuid.uuid4()
    assert jobevents.payload(jid, vid, "transcribing", 40) == {
        "id": str(jid), "video_id": str(vid), "stage": "transcribing", "progress": 40, "error": None}


def test_publish_sends_compact_json_on_the_channel():
    client = MagicMock()
    event = jobevents.payload("j", "v", "done", 100)
    jobevents.publish(event, client=client)
    channel, body = client.publish.call_args[0]
    assert channel == jobevents.CHANNEL and json.loads(body) == event and " " not in body


def test_publish_never_raises_when_redis_is_down(caplog):
    client = MagicMock()
    client.publish.side_effect = redis.ConnectionError("down")
    with caplog.at_level("WARNING"):
        jobevents.publish(jobevents.payload("j", "v", "queued", 0), client=client)  # must not raise: jobs and uploads go on
    assert any("not published" in m for m in caplog.messages)


@pytest.fixture
def env_tasks():
    from app.workers import tasks
    job = SimpleNamespace(stage="queued", progress=0, error=None, video_id=uuid.uuid4())
    db = MagicMock()
    db.get.return_value = job
    with patch.object(tasks.jobevents, "publish") as publish:
        yield SimpleNamespace(db=db, job_id=uuid.uuid4(), video_id=job.video_id, publish=publish)


def test_worker_publishes_every_state_change_after_committing(env_tasks):
    from app.workers import tasks
    order = []
    env_tasks.db.commit.side_effect = lambda: order.append("commit")
    env_tasks.publish.side_effect = lambda e: order.append(("publish", e["stage"], e["progress"]))
    tasks._set(env_tasks.db, str(env_tasks.job_id), "embedding", 70)
    assert order == ["commit", ("publish", "embedding", 70)]
    assert env_tasks.publish.call_args[0][0]["video_id"] == str(env_tasks.video_id)


# ---------------------------------------------------------------- the stream generator
class FakePubSub:
    """Delivers `messages` (bytes, or None = idle timeout), then raises RedisError like a dropped connection."""

    def __init__(self, messages):
        self.messages, self.closed = list(messages), False

    async def subscribe(self, *_):
        pass

    async def get_message(self, ignore_subscribe_messages=False, timeout=None):
        if not self.messages:
            raise RedisError("connection lost")
        item = self.messages.pop(0)
        return None if item is None else {"type": "message", "data": item}

    async def aclose(self):
        self.closed = True


class FakeClient:
    def __init__(self, pubsub):
        self._pubsub, self.closed = pubsub, False

    def pubsub(self):
        return self._pubsub

    async def aclose(self):
        self.closed = True


async def collect(gen):
    return [chunk async for chunk in gen]


def test_stream_sends_snapshot_then_events_then_keepalive_and_cleans_up():
    snap = [jobevents.payload("j1", "v1", "transcribing", 40)]
    ev = json.dumps(jobevents.payload("j1", "v1", "done", 100)).encode()
    pubsub = FakePubSub([ev, None])
    client = FakeClient(pubsub)
    out = asyncio.run(collect(jobs.job_stream(client, pubsub, snap, 0.01)))
    assert out[0] == "event: snapshot\ndata: " + json.dumps({"jobs": snap}) + "\n\n"
    assert out[1] == "event: job\ndata: " + ev.decode() + "\n\n"
    assert out[2] == ": keep-alive\n\n"
    assert len(out) == 3  # the dropped connection ends the stream; the browser reconnects
    assert pubsub.closed and client.closed


def test_stream_closes_redis_when_the_client_disconnects():
    pubsub = FakePubSub([b"{}"] * 5)
    client = FakeClient(pubsub)

    async def go():
        gen = jobs.job_stream(client, pubsub, [], 0.01)
        await gen.__anext__()  # snapshot
        await gen.aclose()  # what Starlette does when the browser goes away

    asyncio.run(go())
    assert pubsub.closed and client.closed


# ---------------------------------------------------------------- the route
@pytest.fixture
def admin_client():
    app.dependency_overrides[current_user] = lambda: CurrentUser(id=str(uuid.uuid4()), role="admin")
    yield TestClient(app)
    app.dependency_overrides.clear()


def override_redis(pubsub):
    app.dependency_overrides[jobs.get_redis_factory] = lambda: (lambda: FakeClient(pubsub))


def test_events_requires_login_and_admin():
    app.dependency_overrides.clear()
    c = TestClient(app)
    assert c.get("/api/jobs/events").status_code == 401  # never 403 for a missing token
    app.dependency_overrides[current_user] = lambda: CurrentUser(id=str(uuid.uuid4()), role="user")
    try:
        assert c.get("/api/jobs/events").status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_events_route_streams_snapshot_and_updates_with_unbuffered_headers(admin_client):
    snap = [jobevents.payload("j1", "v1", "queued", 0)]
    live = json.dumps(jobevents.payload("j1", "v1", "extracting", 10), separators=(",", ":")).encode()
    pubsub = FakePubSub([live])
    override_redis(pubsub)
    with patch.object(jobs, "load_snapshot", return_value=snap):
        r = admin_client.get("/api/jobs/events")  # not read as /api/jobs/{job_id}: no 422 about a UUID
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "no-cache" in r.headers["cache-control"] and r.headers["x-accel-buffering"] == "no"
    assert r.text.index("event: snapshot") < r.text.index("event: job")
    assert '"stage":"extracting"' in r.text
    assert pubsub.closed


def test_events_route_is_a_real_503_when_redis_is_unavailable(admin_client):
    class Broken(FakePubSub):
        async def subscribe(self, *_):
            raise RedisError("down")

    pubsub = Broken([])
    override_redis(pubsub)
    r = admin_client.get("/api/jobs/events")
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]
    assert pubsub.closed


def test_subscribes_before_reading_the_snapshot(admin_client):
    calls = []

    class Recording(FakePubSub):
        async def subscribe(self, channel):
            calls.append(("subscribe", channel))

    override_redis(Recording([]))
    with patch.object(jobs, "load_snapshot", side_effect=lambda: calls.append("snapshot") or []):
        admin_client.get("/api/jobs/events")
    assert calls == [("subscribe", jobevents.CHANNEL), "snapshot"]  # otherwise a change in between would be lost


# ---------------------------------------------------------------- against the real database and Redis
def test_active_jobs_lists_only_unfinished_jobs(api, db_session):
    v = Video(id=uuid.uuid4(), title="t", storage_key="raw/x.mp4")
    db_session.add(v)
    db_session.flush()
    db_session.add_all([Job(video_id=v.id, stage=s, progress=p) for s, p in
                        [("queued", 0), ("transcribing", 40), ("done", 100), ("failed", 0)]])
    db_session.commit()
    snap = jobevents.active_jobs(db_session)
    assert sorted(j["stage"] for j in snap) == ["queued", "transcribing"]
    assert all(j["video_id"] == str(v.id) and set(j) == {"id", "video_id", "stage", "progress", "error"} for j in snap)


def test_publish_reaches_a_real_subscriber(redis_test):
    sub = redis_test.pubsub()
    sub.subscribe(jobevents.CHANNEL)
    sub.get_message(timeout=1)  # the subscribe confirmation
    jobevents.publish(jobevents.payload("j9", "v9", "embedding", 70), client=redis_test)
    msg = sub.get_message(timeout=1)
    assert json.loads(msg["data"])["stage"] == "embedding"
    sub.close()
