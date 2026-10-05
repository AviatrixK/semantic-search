"""Test configuration. Must run before `app.core.config` is imported.

Safety: the app under test is pointed at the `svs_test` database and Redis db 15, never at the dev database or
the dev rate-limiter state. Integration fixtures refuse to run if that is not the case.

Override with TEST_DATABASE_URL / TEST_REDIS_URL. Otherwise they are derived from DATABASE_URL / REDIS_URL
(inside Docker: `docker compose exec api pytest`) with the database name / number swapped.
"""
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

TEST_DB_NAME = "svs_test"
TEST_REDIS_DB = 15


def _swap_path(url: str, path: str) -> str:
    return urlparse(url)._replace(path=path).geturl()


TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL") or _swap_path(
    os.environ.get("DATABASE_URL", "postgresql+psycopg://svs:svs@localhost:5432/svs"), f"/{TEST_DB_NAME}")
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL") or _swap_path(
    os.environ.get("REDIS_URL", "redis://localhost:6379/0"), f"/{TEST_REDIS_DB}")

if not urlparse(TEST_DATABASE_URL).path.lstrip("/").endswith("_test"):
    raise RuntimeError(f"Refusing to run tests: {TEST_DATABASE_URL} is not a *_test database")
if urlparse(TEST_REDIS_URL).path != f"/{TEST_REDIS_DB}":
    raise RuntimeError(f"Refusing to run tests: {TEST_REDIS_URL} must use Redis db {TEST_REDIS_DB}")

os.environ.update({
    "JWT_SECRET": "test-secret-not-for-production-use-0123456789",
    "ACCESS_TOKEN_MINUTES": "15",
    "REFRESH_TOKEN_DAYS": "7",
    "COOKIE_SECURE": "false",
    "DATABASE_URL": TEST_DATABASE_URL,
    "REDIS_URL": TEST_REDIS_URL,
    "S3_ENDPOINT": "http://localhost:9000",
    "S3_PUBLIC_ENDPOINT": "http://localhost:9000",
    "S3_ACCESS_KEY": "test",
    "S3_SECRET_KEY": "test",
    "S3_BUCKET": "test-videos",
    "EMBED_DIM": "384",
    "CHUNK_SECONDS": "30",
    "CHUNK_OVERLAP": "5",
    "MAX_UPLOAD_MB": "1",
    "RATE_LOGIN_PER_MIN": "5",
    "RATE_SEARCH_PER_MIN": "30",
    "RATE_ASK_PER_MIN": "10",
    "MIN_SCORE": "0.25",
    "EMBED_CACHE_TTL_SEC": "86400",
    "LLM_MODEL": "test-model",
    "LLM_TIMEOUT_SEC": "5",
    "LLM_MAX_RETRIES": "2",
    "LLM_MAX_OUTPUT_TOKENS": "256",
    "RAG_TOP_K": "8",
    "AGENT_MAX_STEPS": "6",
    "AGENT_MAX_LLM_CALLS": "8",
    "AGENT_TOOL_RESULT_TOKENS": "1500",
    "DAILY_TOKEN_BUDGET": "50000",
    "CHAT_MEMORY_TURNS": "3",
    "CHAT_MEMORY_TTL_SEC": "3600",
    "SSE_KEEPALIVE_SEC": "0.2",
    "GEMINI_API_KEY": "",
})

# /db/init.sql inside the api container (compose mounts ./db there); <repo>/db/init.sql on the host.
INIT_SQL = Path(__file__).resolve().parents[2] / "db" / "init.sql"
CREATE_HINT = 'docker compose exec postgres psql -U svs -d postgres -P pager=off -c "CREATE DATABASE svs_test;"'


def _unavailable(msg: str):
    """Inside Docker a missing test DB/Redis is a setup error (fail loudly); on a bare dev machine, skip."""
    if os.path.exists("/.dockerenv"):
        pytest.fail(msg, pytrace=False)
    pytest.skip(msg)


@pytest.fixture(scope="session")
def test_db():
    """Rebuilds the svs_test schema from db/init.sql once per session."""
    import psycopg

    from app.core.config import settings
    assert settings.DATABASE_URL == TEST_DATABASE_URL, "app is not pointed at the test database"
    if not INIT_SQL.exists():
        _unavailable(f"{INIT_SQL} not found (compose should mount ./db at /db)")
    try:
        conn = psycopg.connect(TEST_DATABASE_URL.replace("+psycopg", ""), autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as e:
        _unavailable(f"Test database unreachable ({str(e).splitlines()[0]}). Create it with: {CREATE_HINT}")
    with conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        conn.execute(INIT_SQL.read_text(encoding="utf-8"))
    yield TEST_DATABASE_URL


@pytest.fixture
def redis_test():
    """Redis db 15, flushed before and after every test so rate-limiter state never leaks between tests."""
    import redis
    r = redis.Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=2)
    try:
        r.ping()
    except redis.RedisError as e:
        _unavailable(f"Test Redis unreachable ({e})")
    assert r.connection_pool.connection_kwargs["db"] == TEST_REDIS_DB
    r.flushdb()
    yield r
    r.flushdb()


@pytest.fixture
def api(test_db, redis_test):
    """FastAPI TestClient against the real test database, with all tables empty."""
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    from app.core.db import engine
    from app.main import app
    assert engine.url.database == TEST_DB_NAME, f"engine points at {engine.url.database}"
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE users, refresh_tokens, videos, jobs, chunks, search_logs RESTART IDENTITY CASCADE"))
    app.dependency_overrides.clear()
    client = TestClient(app)  # no `with`: the lifespan (storage bucket check) must not run in tests
    yield client
    client.close()


@pytest.fixture
def db_session(api):
    from app.core.db import SessionLocal
    db = SessionLocal()
    yield db
    db.close()
