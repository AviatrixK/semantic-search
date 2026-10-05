"""Short conversation memory for follow-ups ("what about the second video?"): the last few question/answer turns of a chat
session, in Redis, scoped to user AND session id (a client can only ever read its own), expiring on their own.
The server keeps the history, not the client, so nobody can forge earlier turns into the prompt."""
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import redis

from app.core.config import settings

log = logging.getLogger(__name__)
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
MAX_ANSWER_CHARS = 2000


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


@lru_cache
def _redis() -> redis.Redis:
    return redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)


def valid_session_id(session_id: str | None) -> bool:
    return bool(session_id) and SESSION_ID.match(session_id) is not None


class ChatMemory:
    def __init__(self, *, client=None, turns: int | None = None, ttl_sec: int | None = None):
        self._client = client
        self._turns = settings.CHAT_MEMORY_TURNS if turns is None else turns
        self._ttl = settings.CHAT_MEMORY_TTL_SEC if ttl_sec is None else ttl_sec

    def _redis(self):
        return self._client or _redis()

    @staticmethod
    def key(user_id: str, session_id: str) -> str:
        return f"chat:{user_id}:{session_id}"

    def load(self, user_id: str, session_id: str | None) -> list[Turn]:
        """Oldest first. Empty when there is no session, nothing stored, or Redis is down."""
        if not valid_session_id(session_id):
            return []
        try:
            raw = self._redis().lrange(self.key(user_id, session_id), 0, self._turns - 1)
        except redis.RedisError:
            log.warning("chat memory: redis unavailable (read)")
            return []
        turns = []
        for item in reversed(raw):  # stored newest first
            try:
                d = json.loads(item)
                turns.append(Turn(question=str(d["q"]), answer=str(d["a"])))
            except (ValueError, KeyError, TypeError):
                continue
        return turns

    def append(self, user_id: str, session_id: str | None, question: str, answer: str) -> None:
        if not valid_session_id(session_id) or self._turns <= 0:
            return
        try:
            key = self.key(user_id, session_id)
            pipe = self._redis().pipeline()
            pipe.lpush(key, json.dumps({"q": question, "a": answer[:MAX_ANSWER_CHARS]}))
            pipe.ltrim(key, 0, self._turns - 1)
            pipe.expire(key, self._ttl)
            pipe.execute()
        except redis.RedisError:
            log.warning("chat memory: redis unavailable (write)")

    def clear(self, user_id: str, session_id: str | None) -> None:
        if not valid_session_id(session_id):
            return
        try:
            self._redis().delete(self.key(user_id, session_id))
        except redis.RedisError:
            log.warning("chat memory: redis unavailable (clear)")
