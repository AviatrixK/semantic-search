from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg://svs:svs@localhost:5432/svs"
    REDIS_URL: str = "redis://localhost:6379/0"

    JWT_SECRET: str = "change-me"
    ACCESS_TOKEN_MINUTES: int = 15
    REFRESH_TOKEN_DAYS: int = 7
    COOKIE_SECURE: bool = False

    S3_ENDPOINT: str = "http://localhost:9000"
    S3_PUBLIC_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "videos"

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


settings = Settings()
