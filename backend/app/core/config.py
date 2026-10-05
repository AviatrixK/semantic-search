from urllib.parse import unquote, urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Values that must never reach production (checked only when APP_ENV=production).
KNOWN_DEFAULT_JWT_SECRETS = {"", "change-me", "change-me-to-a-long-random-string", "secret", "changeme"}
KNOWN_DEFAULT_DB_PASSWORDS = {"", "svs", "postgres", "password", "change-me", "changeme", "minioadmin"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "development"  # "production" turns on the startup safety checks below and stops bucket auto-creation

    DATABASE_URL: str = "postgresql+psycopg://svs:svs@localhost:5432/svs"
    REDIS_URL: str = "redis://localhost:6379/0"

    JWT_SECRET: str = "change-me"
    ACCESS_TOKEN_MINUTES: int = 15
    REFRESH_TOKEN_DAYS: int = 7
    COOKIE_SECURE: bool = False

    # Unset S3_ENDPOINT = real AWS S3 in AWS_REGION. Unset keys = boto3's default credential chain (EC2 instance role).
    S3_ENDPOINT: str | None = None
    S3_PUBLIC_ENDPOINT: str | None = None  # URL the browser can reach; unset = same as S3_ENDPOINT (AWS S3 when that is unset too)
    S3_ACCESS_KEY: str | None = None
    S3_SECRET_KEY: str | None = None
    S3_BUCKET: str = "videos"
    AWS_REGION: str = "ap-south-1"

    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    WHISPER_SIZE: str = "base"
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIM: int = 384
    CHUNK_SECONDS: float = 30.0
    CHUNK_OVERLAP: float = 5.0
    MAX_UPLOAD_MB: int = 500
    RATE_LOGIN_PER_MIN: int = 5  # per client IP + email
    RATE_SEARCH_PER_MIN: int = 30  # per user
    RATE_ASK_PER_MIN: int = 10  # per user
    MIN_SCORE: float = 0.25  # cosine similarity below this is dropped from search results
    EMBED_CACHE_TTL_SEC: int = 86400
    HYBRID_CANDIDATES: int = 30  # hits taken from EACH of vector and keyword search before they are fused
    RRF_K: int = 60  # Reciprocal Rank Fusion constant: score = sum 1 / (RRF_K + rank)
    RERANK: bool = False  # rescore the best hybrid hits with a cross-encoder (downloads RERANK_MODEL on first use)
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_CANDIDATES: int = 20
    LLM_MODEL: str = "gemini-flash-latest"  # alias for the newest Flash model; pin an exact model id to freeze behaviour
    LLM_TIMEOUT_SEC: float = 30.0  # per attempt
    LLM_MAX_RETRIES: int = 2  # extra attempts on 429/5xx/timeouts (so up to 3 calls)
    LLM_MAX_OUTPUT_TOKENS: int = 2048  # includes thinking tokens on thinking models
    RAG_TOP_K: int = 8  # transcript chunks given to the model
    AGENT_MAX_STEPS: int = 6  # tool-calling rounds before the agent must answer
    AGENT_MAX_LLM_CALLS: int = 8  # hard cap of model calls for ONE request (router + agent steps + forced final)
    AGENT_TOOL_RESULT_TOKENS: int = 1500  # a tool result is cut to about this many tokens before the model sees it
    DAILY_TOKEN_BUDGET: int = 200000  # per user and UTC day; 0 disables the limit
    CHAT_MEMORY_TURNS: int = 3  # question/answer turns remembered per chat session, for follow-ups
    CHAT_MEMORY_TTL_SEC: int = 3600
    SSE_KEEPALIVE_SEC: float = 15.0  # comment line sent on an idle stream so proxies do not close it
    GEMINI_API_KEY: str = ""

    @field_validator("S3_ENDPOINT", "S3_PUBLIC_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY", mode="before")
    @classmethod
    def _blank_is_unset(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.strip().lower() == "production"

    @model_validator(mode="after")
    def _production_guards(self):
        if not self.is_production:
            return self
        problems = []
        if self.JWT_SECRET.strip().lower() in KNOWN_DEFAULT_JWT_SECRETS or len(self.JWT_SECRET) < 32:
            problems.append("JWT_SECRET is a known default or shorter than 32 characters "
                            "(generate one: python -c \"import secrets; print(secrets.token_urlsafe(48))\")")
        password = unquote(urlparse(self.DATABASE_URL).password or "")
        if password.lower() in KNOWN_DEFAULT_DB_PASSWORDS:
            problems.append("the database password in DATABASE_URL is empty or a known default")
        if not self.COOKIE_SECURE:
            problems.append("COOKIE_SECURE must be true (the site is served over HTTPS)")
        if problems:
            raise ValueError("Refusing to start with APP_ENV=production: " + "; ".join(problems))
        return self


settings = Settings()
