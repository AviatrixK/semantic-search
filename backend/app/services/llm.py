"""Thin wrapper over google-genai: timeout, retries on 429/5xx with backoff, token-usage logging, and an interface that
tests (and later the agent) can replace with a fake. Nothing here knows about videos or prompts."""
import logging
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Protocol, TypeVar

from app.core.config import settings

log = logging.getLogger(__name__)
T = TypeVar("T")

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


# ---- errors: each carries a short, user-facing message (never keys, prompts or SDK internals)
class LLMError(Exception):
    user_message = "The answer service is unavailable right now. Please try again in a moment."

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(message)
        if user_message:
            self.user_message = user_message


class LLMNotConfigured(LLMError):
    user_message = "Ask is not set up yet: an administrator needs to configure GEMINI_API_KEY."


class LLMUnavailable(LLMError):
    """Rate limited, overloaded, timed out or unreachable, and it did not clear up after the retries."""
    user_message = "The answer service is busy or unreachable right now. Please try again in a moment."


class LLMRejected(LLMError):
    """A non-retryable 4xx: bad key, unknown model, invalid request."""
    user_message = "The answer service rejected the request. An administrator should check GEMINI_API_KEY and LLM_MODEL."


class LLMEmptyResponse(LLMError):
    user_message = "The model did not return an answer. Please try rephrasing the question."


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    thought_tokens: int | None = None
    total_tokens: int | None = None
    attempts: int = 1
    latency_ms: int = 0


class LLM(Protocol):
    def generate(self, *, system: str, prompt: str) -> LLMResult: ...


# ---- retries
def status_of(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def is_retryable(exc: BaseException) -> bool:
    """429 and 5xx from the API, plus network trouble (timeouts, refused or dropped connections)."""
    if status_of(exc) in RETRYABLE_STATUS:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        import httpx  # the SDK is built on httpx: its timeouts and connection errors are TransportError subclasses
        return isinstance(exc, httpx.TransportError)
    except ImportError:  # pragma: no cover
        return False


def backoff_delay(attempt: int, base: float, cap: float, jitter: float) -> float:
    """Exponential (base, 2x base, 4x base ...) capped, then scaled by 0.5-1.0 so concurrent clients do not retry in step.
    `attempt` counts from 0; `jitter` is a random number in [0, 1)."""
    return min(cap, base * 2 ** attempt) * (0.5 + jitter / 2)


def call_with_retries(
    fn: Callable[[], T], *, max_retries: int, base_delay: float = 1.0, max_delay: float = 10.0,
    sleep: Callable[[float], None] = time.sleep, jitter: Callable[[], float] = random.random,
) -> tuple[T, int]:
    """Runs `fn`; retries retryable errors up to `max_retries` times. Returns (result, attempts used)."""
    for attempt in range(max_retries + 1):
        try:
            return fn(), attempt + 1
        except Exception as exc:
            if attempt == max_retries or not is_retryable(exc):
                raise
            delay = backoff_delay(attempt, base_delay, max_delay, jitter())
            log.warning("llm: retryable error (%s, status=%s), retry %d/%d in %.1fs",
                        type(exc).__name__, status_of(exc), attempt + 1, max_retries, delay)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


# ---- Gemini
def _count(usage, name: str) -> int | None:
    value = getattr(usage, name, None)
    return value if isinstance(value, int) else None


class GeminiLLM:
    def __init__(self, *, model: str, api_key: str, timeout_sec: float, max_retries: int, max_output_tokens: int,
                 client=None, sleep: Callable[[float], None] = time.sleep, jitter: Callable[[], float] = random.random):
        self.model = model
        self._api_key = api_key
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._client = client
        self._sleep = sleep
        self._jitter = jitter

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMNotConfigured("GEMINI_API_KEY is empty")
            from google import genai  # heavy import, only when Ask is actually used
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, *, system: str, prompt: str) -> LLMResult:
        client = self._get_client()
        config = {  # a plain dict is accepted by the SDK, so this module needs no SDK types
            "system_instruction": system,
            "temperature": 0.2,  # answers must stay close to the excerpts
            "max_output_tokens": self._max_output_tokens,  # includes thinking tokens on thinking models
            "http_options": {"timeout": int(self._timeout_sec * 1000)},  # the SDK takes milliseconds
        }
        started = time.monotonic()
        try:
            response, attempts = call_with_retries(
                lambda: client.models.generate_content(model=self.model, contents=prompt, config=config),
                max_retries=self._max_retries, sleep=self._sleep, jitter=self._jitter,
            )
        except LLMError:
            raise
        except Exception as exc:
            status = status_of(exc)
            log.error("llm: call failed model=%s error=%s status=%s message=%.200s", self.model, type(exc).__name__, status, exc)
            if is_retryable(exc):
                raise LLMUnavailable(f"{type(exc).__name__}: status={status}") from exc
            raise LLMRejected(f"{type(exc).__name__}: status={status}") from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = _count(usage, "prompt_token_count")
        completion_tokens = _count(usage, "candidates_token_count")
        thought_tokens = _count(usage, "thoughts_token_count")
        total_tokens = _count(usage, "total_token_count")
        # Token counts only: never the prompt or the answer (they contain user questions and transcript text).
        log.info("llm: ok model=%s attempts=%d latency_ms=%d prompt_tokens=%s completion_tokens=%s thought_tokens=%s total_tokens=%s",
                 self.model, attempts, latency_ms, prompt_tokens, completion_tokens, thought_tokens, total_tokens)

        try:
            text = (getattr(response, "text", None) or "").strip()
        except Exception:  # the SDK can raise when a response has no usable parts
            text = ""
        if not text:
            candidates = getattr(response, "candidates", None) or []
            reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            log.warning("llm: empty response model=%s finish_reason=%s", self.model, reason)
            raise LLMEmptyResponse(f"empty response, finish_reason={reason}")
        return LLMResult(text=text, model=self.model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                         thought_tokens=thought_tokens, total_tokens=total_tokens, attempts=attempts, latency_ms=latency_ms)


# ---- the instance the app uses (replaceable)
_override: LLM | None = None


def set_llm(llm: LLM | None) -> None:
    """Swap in a fake (tests) or restore the real one with None."""
    global _override
    _override = llm


@lru_cache
def _default() -> GeminiLLM:
    return GeminiLLM(model=settings.LLM_MODEL, api_key=settings.GEMINI_API_KEY, timeout_sec=settings.LLM_TIMEOUT_SEC,
                     max_retries=settings.LLM_MAX_RETRIES, max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS)


def get_llm() -> LLM:
    return _override or _default()
