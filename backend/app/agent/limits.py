"""Guardrails around every LLM call of one request: a hard cap on the number of calls, the per-user daily token budget,
and cooperative cancellation. All LLM use (router, RAG, agent) goes through MeteredLLM so none of it can be forgotten."""
import threading
from typing import Callable

from app.services.llm import LLM, LLMResult, Message, ToolSpec, TurnResult
from app.services.usage import DailyBudgetExceeded, UsageMeter, estimate_tokens


class Cancelled(Exception):
    """The client went away: stop spending LLM calls."""


class CallCapExceeded(Exception):
    """This request already used all the LLM calls it is allowed."""


class CallBudget:
    """Counts the LLM calls and tokens of ONE request (thread-safe: tools may run in parallel)."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.calls = 0
        self.tokens = 0
        self._lock = threading.Lock()

    def take(self) -> None:
        with self._lock:
            if self.calls >= self.max_calls:
                raise CallCapExceeded(f"limit of {self.max_calls} model calls per request reached")
            self.calls += 1

    def record(self, tokens: int) -> None:
        with self._lock:
            self.tokens += tokens


class MeteredLLM:
    """Wraps an LLM: before each call it checks cancellation, the daily token budget and the per-request call cap;
    after each call it adds the tokens used to the request total and to the user's daily total."""

    def __init__(self, inner: LLM, budget: CallBudget, *, meter: UsageMeter | None = None, user_id: str | None = None,
                 should_stop: Callable[[], bool] | None = None):
        self._inner = inner
        self.budget = budget
        self._meter = meter
        self._user_id = user_id
        self._should_stop = should_stop

    def _guard(self) -> None:
        if self._should_stop and self._should_stop():
            raise Cancelled()
        if self._meter and self._user_id and self._meter.exceeded(self._user_id):
            raise DailyBudgetExceeded(self._meter.seconds_until_reset())
        self.budget.take()

    def _after(self, total_tokens: int | None, fallback_chars: int) -> None:
        tokens = total_tokens if total_tokens is not None else estimate_tokens("x" * fallback_chars)
        self.budget.record(tokens)
        if self._meter and self._user_id:
            self._meter.add(self._user_id, tokens)

    def generate(self, *, system: str, prompt: str) -> LLMResult:
        self._guard()
        result = self._inner.generate(system=system, prompt=prompt)
        self._after(result.total_tokens, len(system) + len(prompt) + len(result.text))
        return result

    def generate_turn(self, *, system: str, messages: list[Message], tools: list[ToolSpec] | None = None,
                      allow_tools: bool = True) -> TurnResult:
        self._guard()
        turn = self._inner.generate_turn(system=system, messages=messages, tools=tools, allow_tools=allow_tools)
        self._after(turn.total_tokens, len(system) + 600 * len(messages) + len(turn.text or ""))
        return turn
