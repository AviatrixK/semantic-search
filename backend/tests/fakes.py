"""Test doubles for the LLM layer. No SDK, no network, no API key."""
from app.services.llm import LLMResult


class FakeLLM:
    """Records every call and answers with `reply` (a string, or a callable(system, prompt) -> string).
    An Exception instance as `reply` is raised instead."""

    def __init__(self, reply="Fine."):
        self.reply = reply
        self.calls: list[dict] = []

    def generate(self, *, system: str, prompt: str) -> LLMResult:
        self.calls.append({"system": system, "prompt": prompt})
        reply = self.reply(system, prompt) if callable(self.reply) else self.reply
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(text=reply, model="fake", prompt_tokens=1, completion_tokens=1, total_tokens=2)


def hit(n: int = 1, *, video_id: str = "11111111-1111-1111-1111-111111111111", title: str = "How to Speak",
        start: float | None = None, end: float | None = None, text: str | None = None, score: float = 0.5) -> dict:
    """A search hit as retrieval.vector_search returns it."""
    start = 30.0 * n if start is None else start
    return {"chunk_id": f"c{n}", "video_id": video_id, "title": title, "start_sec": start,
            "end_sec": start + 30 if end is None else end, "text": text or f"Chunk number {n} text.", "score": score}
