"""Auth integration tests: real FastAPI app, real Postgres (svs_test) and Redis (db 15). No ML models are loaded."""
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from sqlalchemy import select, text

from app.core.config import settings
from app.models import RefreshToken, User

PW = "Passw0rd123"


def register(api, email="ann@example.com", password=PW, **extra):
    return api.post("/auth/register", json={"email": email, "password": password, **extra})


def login(api, email="ann@example.com", password=PW):
    return api.post("/auth/login", json={"email": email, "password": password})


def bearer(r):
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def refresh_cookie(api):
    return api.cookies.get("refresh_token")


def set_refresh_cookie(api, value):
    api.cookies.clear()
    api.cookies.set("refresh_token", value, domain="testserver.local", path="/auth")


def make_admin(db, email):
    db.execute(text("UPDATE users SET role='admin' WHERE email=:e"), {"e": email})
    db.commit()


# ---- register
def test_register_ok_lowercases_email_and_is_never_admin(api, db_session):
    r = register(api, "Ann@Example.COM", role="admin")  # extra "role" field must be ignored
    assert r.status_code == 201
    assert r.json()["email"] == "ann@example.com" and r.json()["role"] == "user"
    user = db_session.scalar(select(User))
    assert user.password_hash != PW and user.password_hash.startswith("$2")  # bcrypt, not plaintext


def test_register_duplicate_email_409_case_insensitive(api):
    assert register(api).status_code == 201
    assert register(api).status_code == 409
    assert register(api, "ANN@example.com").status_code == 409


@pytest.mark.parametrize("password", ["short1", "allletters", "12345678", "a1" * 40])
def test_register_weak_password_422(api, password):
    assert register(api, password=password).status_code == 422


def test_register_bad_email_422(api):
    assert register(api, "not-an-email").status_code == 422


# ---- login
def test_login_success_returns_token_and_httponly_cookie(api):
    register(api)
    r = login(api)
    assert r.status_code == 200 and r.json()["token_type"] == "bearer"
    set_cookie = r.headers["set-cookie"].lower()
    assert "refresh_token=" in set_cookie and "httponly" in set_cookie and "path=/auth" in set_cookie


def test_login_failures_are_generic_401(api):
    register(api)
    wrong = login(api, password="Wrong1234")
    unknown = login(api, "nobody@example.com")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()  # does not reveal which emails exist


def test_login_is_case_insensitive_on_email(api):
    register(api)
    assert login(api, "ANN@Example.com").status_code == 200


# ---- /auth/me
def test_me_returns_current_user(api):
    register(api)
    r = api.get("/auth/me", headers=bearer(login(api)))
    assert r.status_code == 200 and r.json()["email"] == "ann@example.com"


def test_me_without_or_with_garbage_token_is_401(api):
    assert api.get("/auth/me").status_code == 401
    assert api.get("/auth/me", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_expired_access_token_is_401(api, monkeypatch):
    register(api)
    monkeypatch.setattr(settings, "ACCESS_TOKEN_MINUTES", -1)  # tokens are born expired
    expired = login(api)
    monkeypatch.undo()
    r = api.get("/auth/me", headers=bearer(expired))
    assert r.status_code == 401 and "expired" in r.json()["detail"].lower()


def test_token_signed_with_wrong_secret_is_401(api):
    forged = jwt.encode({"sub": "x", "role": "admin", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
                        "some-other-secret", algorithm="HS256")
    assert api.get("/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_valid_token_for_deleted_user_is_401(api, db_session):
    register(api)
    headers = bearer(login(api))
    db_session.execute(text("DELETE FROM users"))
    db_session.commit()
    assert api.get("/auth/me", headers=headers).status_code == 401


# ---- authorization
def test_user_hitting_admin_routes_is_403(api):
    register(api)
    headers = bearer(login(api))
    upload = api.post("/api/videos", headers=headers, files={"file": ("a.mp4", b"x" * 10, "video/mp4")})
    assert upload.status_code == 403
    assert api.post("/api/videos/00000000-0000-0000-0000-000000000000/reprocess", headers=headers).status_code == 403
    assert api.delete("/api/videos/00000000-0000-0000-0000-000000000000", headers=headers).status_code == 403


def test_admin_passes_the_admin_check(api, db_session):
    register(api)
    make_admin(db_session, "ann@example.com")
    headers = bearer(login(api))  # role is read from the token, so log in after promotion
    r = api.post("/api/videos/00000000-0000-0000-0000-000000000000/reprocess", headers=headers)
    assert r.status_code == 404  # past the 403 gate; the video simply does not exist


# ---- refresh rotation, reuse detection, logout
def test_refresh_rotates_and_old_token_is_rejected(api):
    register(api)
    login(api)
    old = refresh_cookie(api)
    r = api.post("/auth/refresh")
    assert r.status_code == 200 and r.json()["access_token"]
    new = refresh_cookie(api)
    assert new and new != old
    set_refresh_cookie(api, old)
    assert api.post("/auth/refresh").status_code == 401  # single use


def test_reusing_a_revoked_refresh_token_revokes_all_sessions(api, db_session):
    register(api)
    login(api)
    laptop = refresh_cookie(api)
    login(api)  # second session (e.g. phone)
    phone = refresh_cookie(api)
    set_refresh_cookie(api, laptop)
    assert api.post("/auth/refresh").status_code == 200  # legitimate rotation; `laptop` is now revoked
    attacker_replay = laptop
    set_refresh_cookie(api, attacker_replay)
    assert api.post("/auth/refresh").status_code == 401  # replay detected ...
    active = db_session.scalar(text("SELECT count(*) FROM refresh_tokens WHERE NOT revoked"))
    assert active == 0  # ... every token of the user is revoked
    set_refresh_cookie(api, phone)
    assert api.post("/auth/refresh").status_code == 401  # even the untouched second session is dead


def test_reuse_detection_only_affects_the_offending_user(api, db_session):
    register(api, "ann@example.com")
    register(api, "bob@example.com")
    login(api, "bob@example.com")
    bob = refresh_cookie(api)
    login(api, "ann@example.com")
    old = refresh_cookie(api)
    api.post("/auth/refresh")
    set_refresh_cookie(api, old)
    assert api.post("/auth/refresh").status_code == 401
    set_refresh_cookie(api, bob)
    assert api.post("/auth/refresh").status_code == 200


def test_refresh_without_cookie_or_with_unknown_token_is_401(api):
    assert api.post("/auth/refresh").status_code == 401
    set_refresh_cookie(api, "totally-unknown")
    assert api.post("/auth/refresh").status_code == 401


def test_expired_refresh_token_is_401(api, db_session):
    register(api)
    login(api)
    db_session.execute(text("UPDATE refresh_tokens SET expires_at = now() - interval '1 hour'"))
    db_session.commit()
    assert api.post("/auth/refresh").status_code == 401


def test_refresh_token_is_stored_hashed(api, db_session):
    register(api)
    login(api)
    raw = refresh_cookie(api)
    stored = db_session.scalars(select(RefreshToken.token_hash)).all()
    assert len(stored) == 1 and raw not in stored


def test_logout_revokes_the_refresh_token(api, db_session):
    register(api)
    login(api)
    token = refresh_cookie(api)
    assert api.post("/auth/logout").status_code == 204
    assert db_session.scalar(text("SELECT count(*) FROM refresh_tokens WHERE NOT revoked")) == 0
    set_refresh_cookie(api, token)
    assert api.post("/auth/refresh").status_code == 401


def test_logout_without_cookie_is_fine(api):
    assert api.post("/auth/logout").status_code == 204
