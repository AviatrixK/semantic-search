import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.models import RefreshToken, User
from app.services import tokens
from app.workers import tasks
from app.workers.celery_app import celery


def add_token(db, user, *, expires_in, revoked=False):
    t = RefreshToken(user_id=user.id, token_hash=uuid.uuid4().hex, revoked=revoked,
                     expires_at=datetime.now(timezone.utc) + expires_in)
    db.add(t)
    return t


def test_purge_deletes_only_expired_tokens(api, db_session):
    user = User(email="ann@example.com", password_hash="x")
    db_session.add(user)
    db_session.flush()
    expired = add_token(db_session, user, expires_in=timedelta(hours=-1))
    expired_revoked = add_token(db_session, user, expires_in=timedelta(days=-3), revoked=True)
    live = add_token(db_session, user, expires_in=timedelta(days=5))
    live_revoked = add_token(db_session, user, expires_in=timedelta(days=5), revoked=True)  # kept for reuse detection
    db_session.commit()

    assert tokens.purge_expired_refresh_tokens(db_session) == 2
    left = set(db_session.scalars(select(RefreshToken.id)).all())
    assert left == {live.id, live_revoked.id}
    assert expired.id not in left and expired_revoked.id not in left
    assert tokens.purge_expired_refresh_tokens(db_session) == 0  # idempotent


def test_purge_task_runs_against_the_database(api, db_session):
    user = User(email="ann@example.com", password_hash="x")
    db_session.add(user)
    db_session.flush()
    add_token(db_session, user, expires_in=timedelta(hours=-1))
    db_session.commit()
    assert tasks.purge_expired_tokens.run() == 1


def test_purge_is_scheduled_hourly_with_celery_beat():
    entry = celery.conf.beat_schedule["purge-expired-refresh-tokens"]
    assert entry["task"] == tasks.purge_expired_tokens.name and entry["schedule"] == 3600.0
