"""One interface over every embedding provider: local (sentence-transformers), gemini, cohere, openai.

    embedder = get_embedder()                      # the provider chosen by EMBED_PROVIDER
    vectors  = embedder.embed_documents(texts)     # for stored chunks (task type "document")
    vector   = query_vector(embedder, "how do I pause?")   # for a search query (task type "query"): cached, 2 s timeout, 1 retry

Query and document embeddings are NOT the same thing for most providers (Gemini and Cohere embed them differently on purpose), so
the two methods are separate and each provider applies the right task type:
  gemini-embedding-2   no task_type field: the task goes into the text ("task: search result | query: ..." / "title: none | text: ...")
  gemini-embedding-001 task_type RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT
  cohere               input_type search_query / search_document
  openai               symmetric model: one embedding for both (OpenAI's guides define no query/document types)
  local                symmetric (all-MiniLM-L6-v2)

Hosted vectors are L2-normalised here (Gemini-001 and shortened OpenAI/Cohere vectors are not guaranteed to be), so cosine distance
means the same thing for every model. Dimensions: EMBED_DIMENSIONS for hosted providers (pgvector's HNSW index allows at most 2000).

Sources checked 2026-10-18: ai.google.dev/gemini-api/docs/embeddings, docs.cohere.com/reference/embed,
developers.openai.com/api/docs/guides/embeddings."""
import hashlib
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable

import httpx
import numpy as np
import redis

from app.core.config import settings

log = logging.getLogger(__name__)


# ---- errors
class EmbeddingError(Exception):
    user_message = "Indexing the transcript for search failed. Please try again or contact an administrator."


class EmbeddingNotConfigured(EmbeddingError):
    """The chosen provider has no API key (or an unusable setting)."""
    user_message = "Search indexing is not set up: an administrator needs to configure the embedding provider's API key."


class EmbeddingUnavailable(EmbeddingError):
    """Timeout, network error, rate limit or 5xx, still failing after the retries. Search then falls back to keyword-only."""
    user_message = "The embedding service was unreachable. Please try again in a moment."


class EmbeddingRejected(EmbeddingError):
    """A non-retryable 4xx: bad key, unknown model, input too long."""
    user_message = "The embedding service rejected the request. An administrator should check the API key and EMBED_PROVIDER settings."


# ---- helpers
def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else list(vector)


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English, fewer per token for Indic scripts, so this under-counts them).
    Used only when a provider does not report usage."""
    return max(1, len(text) // 4)


def retryable_status(code: int) -> bool:
    return code == 429 or code >= 500


class HostedEmbedder:
    """Shared plumbing of the hosted providers: an httpx client (injectable for tests), timeouts, retries, token counting."""
    provider = ""
    uses_legacy_column = False  # vectors live in chunk_embeddings, not in chunks.embedding
    supports_sentence_vectors = False

    def __init__(self, *, model: str, dim: int, api_key: str, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.model = model
        self.dim = dim
        self._api_key = api_key
        self._client = client
        self._sleep = sleep
        self.tokens_used = 0  # running total of input tokens: the provider's own count where its response has one, else an estimate
        self.tokens_estimated = 0  # the part of tokens_used that was ESTIMATED because the response carried no usage figure

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}@{self.dim}"

    def check_configured(self) -> None:
        if not self._api_key:
            raise EmbeddingNotConfigured(f"{self.provider.upper()}_API_KEY is empty (EMBED_PROVIDER={self.provider})")

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def _count_tokens(self, reported, texts: list[str]) -> None:
        """Adds the provider's own token count when the response had one (an int), and only otherwise an estimate of `texts`."""
        if isinstance(reported, int) and not isinstance(reported, bool):
            self.tokens_used += reported
        else:
            guess = sum(estimate_tokens(t) for t in texts)
            self.tokens_used += guess
            self.tokens_estimated += guess

    def _post(self, url: str, *, json: dict, headers: dict, timeout: float, retries: int) -> dict:
        """POST with retries on timeouts, network errors, 429 and 5xx. Returns the parsed JSON body."""
        self.check_configured()
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = self._http().post(url, json=json, headers=headers, timeout=timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                log.warning("embedding %s: %s (attempt %d/%d)", self.key, type(exc).__name__, attempt + 1, retries + 1)
            else:
                if r.status_code < 400:
                    return r.json()
                if not retryable_status(r.status_code):
                    raise EmbeddingRejected(f"{self.provider} answered {r.status_code}")
                last = EmbeddingUnavailable(f"{self.provider} answered {r.status_code}")
                log.warning("embedding %s: HTTP %d (attempt %d/%d)", self.key, r.status_code, attempt + 1, retries + 1)
            if attempt < retries:
                self._sleep(min(2.0, 0.3 * 2 ** attempt))
        raise EmbeddingUnavailable(f"{self.provider} unavailable: {type(last).__name__}") from last

    # subclasses
    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover - interface
        raise NotImplementedError

    def _query_args(self) -> dict:
        return {"timeout": settings.EMBED_QUERY_TIMEOUT_SEC, "retries": settings.EMBED_QUERY_RETRIES}

    def _doc_args(self) -> dict:
        return {"timeout": settings.EMBED_DOC_TIMEOUT_SEC, "retries": 3}


# ---- Gemini
class GeminiEmbedder(HostedEmbedder):
    provider = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

    @property
    def _prefix_style(self) -> bool:
        """gemini-embedding-2 has no task_type: the task is written into the text. Older models (-001) take task_type."""
        return not self.model.endswith("-001") and "embedding-001" not in self.model

    def _body(self, text: str, query: bool) -> dict:
        if self._prefix_style:
            text = f"task: search result | query: {text}" if query else f"title: none | text: {text}"
            body: dict = {"content": {"parts": [{"text": text}]}}
        else:
            body = {"content": {"parts": [{"text": text}]}, "taskType": "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"}
        body["outputDimensionality"] = self.dim
        return body

    def _embed_one(self, text: str, query: bool, args: dict) -> list[float]:
        data = self._post(self.URL.format(model=self.model), json=self._body(text, query),
                          headers={"x-goog-api-key": self._api_key}, **args)
        try:
            values = data["embedding"]["values"]
        except (KeyError, TypeError) as exc:
            raise EmbeddingRejected("unexpected Gemini response shape") from exc
        self._count_tokens((data.get("usageMetadata") or {}).get("promptTokenCount"), [text])
        return normalize(values)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text, True, self._query_args())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        args = self._doc_args()
        with ThreadPoolExecutor(max_workers=6) as pool:  # one request per text: the batch endpoint is not documented for embedding-2
            return list(pool.map(lambda t: self._embed_one(t, False, args), texts))


# ---- Cohere
class CohereEmbedder(HostedEmbedder):
    provider = "cohere"
    URL = "https://api.cohere.com/v2/embed"
    BATCH = 96  # the documented maximum number of texts per request

    def _call(self, texts: list[str], input_type: str, args: dict) -> list[list[float]]:
        body = {"model": self.model, "input_type": input_type, "texts": texts, "embedding_types": ["float"], "truncate": "END"}
        if self.model.startswith(("embed-v4", "embed-v5")):  # output_dimension exists only for v4 and newer
            body["output_dimension"] = self.dim
        data = self._post(self.URL, json=body, headers={"Authorization": f"Bearer {self._api_key}"}, **args)
        try:
            vectors = data["embeddings"]["float"]
        except (KeyError, TypeError) as exc:
            raise EmbeddingRejected("unexpected Cohere response shape") from exc
        billed = ((data.get("meta") or {}).get("billed_units") or {}).get("input_tokens")
        self._count_tokens(billed, texts)
        return [normalize(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], "search_query", self._query_args())[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            out += self._call(texts[i:i + self.BATCH], "search_document", self._doc_args())
        return out


# ---- OpenAI
class OpenAIEmbedder(HostedEmbedder):
    provider = "openai"
    URL = "https://api.openai.com/v1/embeddings"
    BATCH = 64

    def _call(self, texts: list[str], args: dict) -> list[list[float]]:
        body = {"model": self.model, "input": texts, "dimensions": self.dim, "encoding_format": "float"}
        data = self._post(self.URL, json=body, headers={"Authorization": f"Bearer {self._api_key}"}, **args)
        try:
            rows = sorted(data["data"], key=lambda d: d["index"])
            vectors = [d["embedding"] for d in rows]
        except (KeyError, TypeError) as exc:
            raise EmbeddingRejected("unexpected OpenAI response shape") from exc
        used = (data.get("usage") or {}).get("total_tokens")
        self._count_tokens(used, texts)
        return [normalize(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], self._query_args())[0]  # symmetric model: no query/document distinction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            out += self._call(texts[i:i + self.BATCH], self._doc_args())
        return out


# ---- local
class LocalEmbedder:
    """The sentence-transformers model that has always been used. Vectors stay in chunks.embedding (384 dimensions)."""
    provider = "local"
    uses_legacy_column = True
    supports_sentence_vectors = True
    tokens_used = 0
    tokens_estimated = 0

    def __init__(self):
        self.model = settings.EMBED_MODEL
        self.dim = settings.EMBED_DIM

    @property
    def key(self) -> str:
        return f"local/{self.model}@{self.dim}"

    def check_configured(self) -> None:
        return None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from app.services import embedding
        return embedding.embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


Embedder = LocalEmbedder | GeminiEmbedder | CohereEmbedder | OpenAIEmbedder

_KEYS = {"gemini": lambda: settings.GEMINI_API_KEY, "cohere": lambda: settings.COHERE_API_KEY, "openai": lambda: settings.OPENAI_API_KEY}
_MODELS = {"gemini": lambda: settings.GEMINI_EMBED_MODEL, "cohere": lambda: settings.COHERE_EMBED_MODEL,
           "openai": lambda: settings.OPENAI_EMBED_MODEL}
_CLASSES = {"gemini": GeminiEmbedder, "cohere": CohereEmbedder, "openai": OpenAIEmbedder}
PROVIDERS = ("local", "gemini", "cohere", "openai")


def make_embedder(provider: str, *, model: str | None = None, dim: int | None = None, client: httpx.Client | None = None):
    if provider == "local":
        return LocalEmbedder()
    if provider not in _CLASSES:
        raise ValueError(f"unknown embedding provider {provider!r}; use one of {', '.join(PROVIDERS)}")
    return _CLASSES[provider](model=model or _MODELS[provider](), dim=dim or settings.EMBED_DIMENSIONS,
                              api_key=_KEYS[provider](), client=client)


@lru_cache
def _default(provider: str, model: str, dim: int):
    return make_embedder(provider, model=model or None, dim=dim)


def get_embedder(provider: str | None = None):
    """The embedder for EMBED_PROVIDER (or an explicit provider), built once per process."""
    provider = provider or settings.EMBED_PROVIDER
    if provider == "local":
        return LocalEmbedder()
    return _default(provider, _MODELS[provider](), settings.EMBED_DIMENSIONS)


# ---- query vectors: cache + resilience
@lru_cache
def _redis() -> redis.Redis:
    return redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)


def cache_key(embedder, query: str) -> str:
    """"emb:" + sha256(provider + model + dimensions + normalised query). Whitespace is normalised so "a  b " and "a b" share an
    entry; vectors from different providers, models or sizes can never be mixed up."""
    normalised = " ".join(query.split())
    return "emb:" + hashlib.sha256(f"{embedder.provider}\n{embedder.model}\n{embedder.dim}\n{normalised}".encode()).hexdigest()


def query_vector(embedder, text: str, *, client=None) -> list[float]:
    """The embedding of a search query, cached in Redis for EMBED_CACHE_TTL_SEC (7 days). Redis trouble never breaks search: on any
    Redis error the vector is simply computed. Hosted providers get EMBED_QUERY_TIMEOUT_SEC and EMBED_QUERY_RETRIES; if they still
    fail, EmbeddingUnavailable is raised and the caller falls back to keyword-only search."""
    key = cache_key(embedder, text)
    r = None
    try:
        r = client or _redis()
        raw = r.get(key)
        if raw and len(raw) == 4 * embedder.dim:
            return np.frombuffer(raw, dtype=np.float32).tolist()
    except redis.RedisError:
        log.warning("embedding cache unavailable (read)")
        r = None
    vec = embedder.embed_query(" ".join(text.split()))
    if r is not None:
        try:
            r.set(key, np.asarray(vec, dtype=np.float32).tobytes(), ex=settings.EMBED_CACHE_TTL_SEC)
        except redis.RedisError:
            log.warning("embedding cache unavailable (write)")
    return vec
