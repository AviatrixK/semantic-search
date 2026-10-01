"""Safe defaults so unit tests run without a .env. Must execute before `app.core.config` is imported."""
import os

os.environ.update({
    "JWT_SECRET": "test-secret-not-for-production-use-0123456789",
    "ACCESS_TOKEN_MINUTES": "15",
    "REFRESH_TOKEN_DAYS": "7",
    "COOKIE_SECURE": "false",
    "DATABASE_URL": "postgresql+psycopg://test:test@localhost:5432/test",
    "REDIS_URL": "redis://localhost:6379/15",
    "S3_ENDPOINT": "http://localhost:9000",
    "S3_PUBLIC_ENDPOINT": "http://localhost:9000",
    "S3_ACCESS_KEY": "test",
    "S3_SECRET_KEY": "test",
    "S3_BUCKET": "test-videos",
    "EMBED_DIM": "384",
    "CHUNK_SECONDS": "30",
    "CHUNK_OVERLAP": "5",
    "GEMINI_API_KEY": "",
})
