from typing import Literal
from urllib.parse import unquote, urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Values that must never reach production (checked only when APP_ENV=production).
KNOWN_DEFAULT_JWT_SECRETS = {"", "change-me", "change-me-to-a-long-random-string", "secret", "changeme"}
KNOWN_DEFAULT_DB_PASSWORDS = {"", "svs", "postgres", "password", "change-me", "changeme", "minioadmin"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LOG_LEVEL: str = "INFO"  # DEBUG | INFO | WARNING | ERROR | CRITICAL; applied by app/core/logging.py (api, worker, beat)

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

    # ---- Transcription: "whisper" = local faster-whisper (default), "sarvam" = Sarvam Saaras via its hosted batch API.
    TRANSCRIBER: Literal["whisper", "sarvam"] = "whisper"
    SARVAM_API_KEY: str = ""
    SARVAM_BASE_URL: str = "https://api.sarvam.ai"
    # saaras:v4 is Sarvam's current model, saaras:v3 the previous one. Languages in SARVAM_V4_UNSUPPORTED (none, per the docs) and any
    # language the v4 job request is rejected for (HTTP 400/422) are transcribed with v3 automatically.
    SARVAM_MODEL: str = "saaras:v3"
    SARVAM_FALLBACK_MODEL: str = "saaras:v3"
    SARVAM_V4_UNSUPPORTED: str = ""  # comma-separated language codes known not to work on v4, e.g. "sat-IN,mni-IN"
    # The transcript is fetched ONCE (mode "transcribe", native script). text_roman and text_en are made per chunk from that native text
    # with Sarvam's text APIs, which bill per character (INR 20 per 10K characters each), on top of the INR 30/h of the transcription.
    # Comma-separated subset of: roman (POST /transliterate), en (POST /translate). Leave one out to save its share of the cost.
    SARVAM_TEXT_VARIANTS: str = "roman,en"
    SARVAM_TRANSLATE_MODEL: str = "sarvam-translate:v1"  # 23 languages, 2000 characters per request; "mayura:v1": 11 languages, 1000
    SARVAM_TRANSLIT_MAX_CHARS: int = 1000  # the docs give no limit for /transliterate: stay well below any plausible one
    SARVAM_TEXT_TIMEOUT_SEC: float = 30.0
    # ---- Who makes text_en (translation) and text_roman (romanisation) of each chunk after Sarvam has transcribed (TRANSCRIBER=sarvam):
    # gemini = a Gemini Flash-Lite model, several chunks per request (needs GEMINI_API_KEY); sarvam = Sarvam's text APIs (SARVAM_TEXT_VARIANTS
    # above); none = neither (native text only). Transcripts are SENT TO GOOGLE when this is gemini.
    TEXT_VARIANT_PROVIDER: Literal["gemini", "sarvam", "none"] = "gemini"
    TEXT_VARIANT_MODEL: str = "gemini-3.1-flash-lite"  # pinned on purpose: an alias such as "-latest" is refused, it could change under us
    TEXT_VARIANT_BATCH_ITEMS: int = 15  # chunks per request, at most
    TEXT_VARIANT_BATCH_CHARS: int = 6000  # and at most this many characters of chunk text per request
    TEXT_VARIANT_TEMPERATURE: float = 0.0  # Google advises 1.0 for Gemini 3; 0 is what was asked for, and every answer is validated anyway
    TEXT_VARIANT_THINKING: str = "minimal"  # generationConfig.thinkingConfig.thinkingLevel; empty = do not send it (thinking tokens are billed as output)
    TEXT_VARIANT_MAX_OUTPUT_TOKENS: int = 16384
    TEXT_VARIANT_TIMEOUT_SEC: float = 60.0
    TEXT_VARIANT_RETRIES: int = 4  # extra attempts after 429 / 5xx / timeout, with exponential backoff (1, 2, 4, 8 s)
    SARVAM_POLL_SEC: float = 5.0
    SARVAM_JOB_TIMEOUT_SEC: int = 3 * 3600  # give up waiting for a batch job after this long
    SARVAM_MAX_FILE_SEC: int = 6600  # batch limit is 2 h per file: longer audio is split into parts of at most this length

    WHISPER_SIZE: str = "base"
    # CPU threads the worker may use for transcription (faster-whisper cpu_threads) and for torch (embeddings). 1 keeps one of
    # the production server's two CPUs free for the API while a video is processed. Raise on a bigger machine.
    WHISPER_THREADS: int = 1
    TORCH_THREADS: int = 1
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIM: int = 384  # dimension of the LOCAL model's vectors (chunks.embedding, chunk_sentences); changing it needs a schema change
    # ---- Embeddings: "local" = the sentence-transformers model above (default); gemini | cohere | openai = hosted.
    EMBED_PROVIDER: Literal["local", "gemini", "cohere", "openai"] = "local"
    EMBED_DIMENSIONS: int = 1536  # vector size for the HOSTED providers (stored in chunk_embeddings); pgvector's HNSW limit is 2000
    GEMINI_API_KEY: str = ""
    GEMINI_EMBED_MODEL: str = "gemini-embedding-2"  # "gemini-embedding-001" also works (task_type instead of text prefixes)
    COHERE_API_KEY: str = ""
    COHERE_EMBED_MODEL: str = "embed-v5.0-pro"
    OPENAI_API_KEY: str = ""
    OPENAI_EMBED_MODEL: str = "text-embedding-3-large"
    EMBED_QUERY_TIMEOUT_SEC: float = 2.0  # one search query's embedding call; on failure search falls back to keyword-only
    EMBED_QUERY_RETRIES: int = 1  # extra attempts after a timeout / 5xx / network error
    EMBED_DOC_TIMEOUT_SEC: float = 30.0  # per request while embedding a video's chunks (no latency pressure there)
    WARMUP_ON_START: bool = True  # api: load the embedding model and encode once at startup, before the app reports ready
    CHUNK_SECONDS: float = 30.0
    CHUNK_OVERLAP: float = 5.0
    MAX_UPLOAD_MB: int = 500
    RATE_LOGIN_PER_MIN: int = 5  # per client IP + email
    RATE_SEARCH_PER_MIN: int = 30  # per user
    MIN_SCORE: float = 0.25  # cosine similarity below this is dropped from search results (for providers with no value of their own)
    # Cosine similarities are spread differently by every embedding model, so each provider can have its own cut-off. Unset = MIN_SCORE.
    # 0.25 was tuned for the local MiniLM model; the hosted values are NOT measured yet: tune them with eval/run_provider_eval.py.
    MIN_SCORE_LOCAL: float | None = None
    MIN_SCORE_GEMINI: float | None = None
    MIN_SCORE_COHERE: float | None = None
    MIN_SCORE_OPENAI: float | None = None
    EMBED_CACHE_TTL_SEC: int = 7 * 86400  # cached query embeddings (key = provider + model + dimensions + normalised query)
    HYBRID_CANDIDATES: int = 30  # hits taken from EACH of vector and keyword search before they are fused
    RRF_K: int = 60  # Reciprocal Rank Fusion constant: score = sum 1 / (RRF_K + rank)
    RERANK: bool = False  # rescore the best hybrid hits with a cross-encoder (downloads RERANK_MODEL on first use)
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_CANDIDATES: int = 20
    SSE_KEEPALIVE_SEC: float = 15.0  # comment line sent on an idle stream (GET /api/jobs/events) so proxies do not close it

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def _log_level(cls, v):
        name = str(v).strip().upper()
        if name not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL (got {v!r})")
        return name

    @field_validator("TEXT_VARIANT_MODEL")
    @classmethod
    def _pinned_text_model(cls, v: str) -> str:
        v = v.strip()
        if not v or v.endswith("-latest") or "latest" in v.split("-"):
            raise ValueError("TEXT_VARIANT_MODEL must be a pinned model id such as gemini-3.1-flash-lite (no -latest alias)")
        return v.removeprefix("models/")

    @field_validator("TEXT_VARIANT_BATCH_ITEMS", "TEXT_VARIANT_BATCH_CHARS", "TEXT_VARIANT_MAX_OUTPUT_TOKENS")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be at least 1")
        return v

    @field_validator("EMBED_DIMENSIONS")
    @classmethod
    def _embed_dimensions(cls, v: int) -> int:
        if not 64 <= v <= 2000:
            raise ValueError("EMBED_DIMENSIONS must be between 64 and 2000 (pgvector's HNSW index supports at most 2000 dimensions)")
        return v

    @field_validator("S3_ENDPOINT", "S3_PUBLIC_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY", mode="before")
    @classmethod
    def _blank_is_unset(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("MIN_SCORE_LOCAL", "MIN_SCORE_GEMINI", "MIN_SCORE_COHERE", "MIN_SCORE_OPENAI", mode="before")
    @classmethod
    def _blank_score_is_unset(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    def min_score_for(self, provider: str) -> float:
        """The similarity cut-off of an embedding provider: its own MIN_SCORE_<PROVIDER> if set, else MIN_SCORE."""
        own = getattr(self, f"MIN_SCORE_{provider.upper()}", None)
        return self.MIN_SCORE if own is None else own

    @property
    def sarvam_text_variants(self) -> list[str]:
        """SARVAM_TEXT_VARIANTS as a clean list drawn from roman, en (in that order, no duplicates)."""
        wanted = [v.strip().lower() for v in self.SARVAM_TEXT_VARIANTS.split(",") if v.strip()]
        unknown = [v for v in wanted if v not in ("roman", "en")]
        if unknown:
            raise ValueError(f"SARVAM_TEXT_VARIANTS: unknown variant(s) {unknown}; use roman, en")
        return [v for v in ("roman", "en") if v in wanted]

    @property
    def sarvam_v4_unsupported(self) -> frozenset:
        return frozenset(c.strip() for c in self.SARVAM_V4_UNSUPPORTED.split(",") if c.strip())

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
