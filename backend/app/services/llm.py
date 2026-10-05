"""Thin wrapper over google-genai: timeout, retries on 429/5xx with backoff, token-usage logging, and an interface that
tests (and the agent) can replace with a fake. Nothing here knows about videos or prompts.

Two calls: generate() is one question in, one text out (RAG, router). generate_turn() is one step of a tool-calling
conversation: provider-neutral messages and tool specs in, either text or tool calls out."""
import logging
import random
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Protocol, TypeVar

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


# ---- results and provider-neutral conversation types
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


@dataclass(frozen=True)
class ToolSpec:
    """A function the model may call. `parameters` is a plain JSON Schema object."""
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    id: str | None = None


@dataclass(frozen=True)
class ToolResponse:
    """The answer to one ToolCall (same name/id). `content` must be a JSON-able dict."""
    name: str
    content: dict
    id: str | None = None


@dataclass(frozen=True)
class UserMsg:
    text: str


@dataclass(frozen=True)
class ModelMsg:
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    # The provider's own object for this turn. It is sent back unchanged: Gemini thinking models attach "thought
    # signatures" to function calls and require them in the history of the next request.
    raw: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ToolMsg:
    """All results for the tool calls of ONE model turn, in the same order as the calls."""
    responses: tuple[ToolResponse, ...]


Message = UserMsg | ModelMsg | ToolMsg


@dataclass(frozen=True)
class TurnResult:
    text: str | None
    tool_calls: tuple[ToolCall, ...]
    raw: Any = field(default=None, compare=False, repr=False)
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    thought_tokens: int | None = None
    total_tokens: int | None = None
    attempts: int = 1
    latency_ms: int = 0


class LLM(Protocol):
    def generate(self, *, system: str, prompt: str) -> LLMResult: ...

    def generate_turn(self, *, system: str, messages: list[Message], tools: list[ToolSpec] | None = None,
                      allow_tools: bool = True) -> TurnResult: ...


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

    def _base_config(self, system: str) -> dict:
        return {  # a plain dict is accepted by the SDK, so this module needs no SDK types
            "system_instruction": system,
            "temperature": 0.2,  # answers must stay close to the excerpts
            "max_output_tokens": self._max_output_tokens,  # includes thinking tokens on thinking models
            "http_options": {"timeout": int(self._timeout_sec * 1000)},  # the SDK takes milliseconds
        }

    def _invoke(self, contents, config: dict):
        """One model call with retries and usage logging. Returns (response, attempts, latency_ms, usage tuple)."""
        client = self._get_client()
        started = time.monotonic()
        try:
            response, attempts = call_with_retries(
                lambda: client.models.generate_content(model=self.model, contents=contents, config=config),
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
        counts = (_count(usage, "prompt_token_count"), _count(usage, "candidates_token_count"),
                  _count(usage, "thoughts_token_count"), _count(usage, "total_token_count"))
        # Token counts only: never the prompt or the answer (they contain user questions and transcript text).
        log.info("llm: ok model=%s attempts=%d latency_ms=%d prompt_tokens=%s completion_tokens=%s thought_tokens=%s total_tokens=%s",
                 self.model, attempts, latency_ms, *counts)
        return response, attempts, latency_ms, counts

    def _empty(self, response) -> LLMEmptyResponse:
        candidates = getattr(response, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        log.warning("llm: empty response model=%s finish_reason=%s", self.model, reason)
        return LLMEmptyResponse(f"empty response, finish_reason={reason}")

    def generate(self, *, system: str, prompt: str) -> LLMResult:
        response, attempts, latency_ms, (p, c, th, tot) = self._invoke(prompt, self._base_config(system))
        try:
            text = (getattr(response, "text", None) or "").strip()
        except Exception:  # the SDK can raise when a response has no usable parts
            text = ""
        if not text:
            raise self._empty(response)
        return LLMResult(text=text, model=self.model, prompt_tokens=p, completion_tokens=c, thought_tokens=th,
                         total_tokens=tot, attempts=attempts, latency_ms=latency_ms)

    # -- tool calling
    @staticmethod
    def _to_content(m: Message):
        if isinstance(m, UserMsg):
            return {"role": "user", "parts": [{"text": m.text}]}
        if isinstance(m, ModelMsg):
            if m.raw is not None:
                return m.raw  # echo the model's own turn untouched (keeps thought signatures)
            parts: list[dict] = [{"text": m.text}] if m.text else []
            for c in m.tool_calls:
                call = {"name": c.name, "args": c.args}
                if c.id:
                    call["id"] = c.id
                parts.append({"function_call": call})
            return {"role": "model", "parts": parts}
        parts = []
        for r in m.responses:
            resp = {"name": r.name, "response": r.content}
            if r.id:
                resp["id"] = r.id
            parts.append({"function_response": resp})
        return {"role": "user", "parts": parts}  # Gemini expects function results in a user turn

    @staticmethod
    def _parse_turn(response) -> tuple[str | None, tuple[ToolCall, ...], Any]:
        candidates = getattr(response, "candidates", None) or []
        content = candidates[0].content if candidates else None
        calls: list[ToolCall] = []
        texts: list[str] = []
        for part in (getattr(content, "parts", None) or []):
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                calls.append(ToolCall(name=fc.name, args=dict(getattr(fc, "args", None) or {}), id=getattr(fc, "id", None)))
            elif getattr(part, "text", None) and not getattr(part, "thought", False):
                texts.append(part.text)
        text = "".join(texts).strip() or None
        return text, tuple(calls), content

    def generate_turn(self, *, system: str, messages: list[Message], tools: list[ToolSpec] | None = None,
                      allow_tools: bool = True) -> TurnResult:
        config = self._base_config(system)
        if tools:
            config["tools"] = [{"function_declarations": [
                {"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]}]  # plain OpenAPI-style schema
            config["automatic_function_calling"] = {"disable": True}  # WE run the tools, so we can trace, cap and validate them
            if not allow_tools:  # tools stay declared (the history contains calls) but the model must answer in text
                config["tool_config"] = {"function_calling_config": {"mode": "NONE"}}
        response, attempts, latency_ms, (p, c, th, tot) = self._invoke([self._to_content(m) for m in messages], config)
        text, calls, raw = self._parse_turn(response)
        if not text and not calls:
            raise self._empty(response)
        return TurnResult(text=text, tool_calls=calls, raw=raw, model=self.model, prompt_tokens=p, completion_tokens=c,
                          thought_tokens=th, total_tokens=tot, attempts=attempts, latency_ms=latency_ms)


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
