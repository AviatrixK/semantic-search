"""Test doubles for the LLM layer and Redis. No SDK, no network, no API key."""
import fnmatch
from typing import Callable

import redis

from app.services.llm import LLMResult, ToolCall, TurnResult


class FakeLLM:
    """Records every call and answers with `reply` (a string, or a callable(system, prompt) -> string).
    An Exception instance as `reply` is raised instead. (generate() only: RAG and the router.)"""

    def __init__(self, reply="Fine."):
        self.reply = reply
        self.calls: list[dict] = []

    def generate(self, *, system: str, prompt: str) -> LLMResult:
        self.calls.append({"system": system, "prompt": prompt})
        reply = self.reply(system, prompt) if callable(self.reply) else self.reply
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(text=reply, model="fake", prompt_tokens=1, completion_tokens=1, total_tokens=2)


class ScriptedLLM(FakeLLM):
    """An LLM that plays a script of model turns for the agent loop. Each item is a TurnResult, an Exception to raise, or a
    callable(messages) -> TurnResult/Exception (so a turn can depend on what the tools returned). Records what it was sent."""

    def __init__(self, turns=(), reply="Fine."):
        super().__init__(reply)
        self.turns = list(turns)
        self.turn_calls: list[dict] = []

    def generate_turn(self, *, system, messages, tools=None, allow_tools=True) -> TurnResult:
        self.turn_calls.append({"system": system, "messages": list(messages), "tools": tools, "allow_tools": allow_tools})
        if not self.turns:
            raise AssertionError("ScriptedLLM ran out of scripted turns")
        item = self.turns.pop(0)
        if callable(item):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        return item


def call(name: str, **args) -> ToolCall:
    return ToolCall(name=name, args=args)


def tools_turn(*calls: ToolCall, text: str | None = None, total_tokens: int = 100) -> TurnResult:
    """A model turn that asks for tools."""
    return TurnResult(text=text, tool_calls=tuple(calls), model="fake", total_tokens=total_tokens)


def text_turn(text: str, total_tokens: int = 100) -> TurnResult:
    """A model turn that answers."""
    return TurnResult(text=text, tool_calls=(), model="fake", total_tokens=total_tokens)


def hit(n: int = 1, *, video_id: str = "11111111-1111-1111-1111-111111111111", title: str = "How to Speak",
        start: float | None = None, end: float | None = None, text: str | None = None, score: float = 0.5) -> dict:
    """A search hit as retrieval.vector_search returns it."""
    start = 30.0 * n if start is None else start
    return {"chunk_id": f"c{n}", "video_id": video_id, "title": title, "start_sec": start,
            "end_sec": start + 30 if end is None else end, "text": text or f"Chunk number {n} text.", "score": score,
            "highlight": None}


class FakeRedis:
    """The few Redis commands the app uses (strings with INCRBY, lists, expiry, pipelines), in memory.
    Set `down = True` to make every command raise like a dead server."""

    def __init__(self):
        self.data: dict = {}
        self.ttl: dict = {}
        self.down = False

    def _check(self):
        if self.down:
            raise redis.ConnectionError("redis is down")

    def get(self, key):
        self._check()
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self._check()
        self.data[key] = value
        if ex:
            self.ttl[key] = ex

    def incrby(self, key, amount):
        self._check()
        self.data[key] = int(self.data.get(key, 0)) + amount
        return self.data[key]

    def expire(self, key, seconds):
        self._check()
        self.ttl[key] = seconds
        return True

    def lpush(self, key, value):
        self._check()
        self.data.setdefault(key, []).insert(0, value)

    def ltrim(self, key, start, stop):
        self._check()
        self.data[key] = self.data.get(key, [])[start: stop + 1]

    def lrange(self, key, start, stop):
        self._check()
        return list(self.data.get(key, [])[start: stop + 1])

    def delete(self, *keys):
        self._check()
        for k in keys:
            self.data.pop(k, None)

    def keys(self, pattern="*"):
        return [k for k in self.data if fnmatch.fnmatch(k, pattern)]

    def pipeline(self):
        outer = self

        class Pipe:
            def __init__(self):
                self.ops: list[Callable] = []

            def __getattr__(self, name):
                def queue(*a, **kw):
                    self.ops.append(lambda: getattr(outer, name)(*a, **kw))
                    return self
                return queue

            def execute(self):
                return [op() for op in self.ops]

        return Pipe()
