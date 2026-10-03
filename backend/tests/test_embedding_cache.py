"""Query-embedding cache with a fake Redis. The real embedding model is never loaded (embed_batch is patched)."""
from unittest.mock import patch

import redis

from app.core.config import settings
from app.services import embedding

VEC = [0.5] * settings.EMBED_DIM  # exactly representable in float32, so the round trip is lossless


class FakeRedis:
    def __init__(self):
        self.data, self.ttl = {}, {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key], self.ttl[key] = value, ex


def test_second_lookup_is_served_from_cache():
    r = FakeRedis()
    with patch.object(embedding, "embed_batch", return_value=[VEC]) as model:
        first = embedding.embed_query("how to speak well", client=r)
        second = embedding.embed_query("how to speak well", client=r)
    assert first == second == VEC
    assert model.call_count == 1
    (key,) = r.data
    assert key.startswith("emb:") and len(key) == 4 + 64  # sha256 hex
    assert r.ttl[key] == 86400  # one day


def test_whitespace_variants_share_an_entry_but_other_queries_do_not():
    r = FakeRedis()
    with patch.object(embedding, "embed_batch", return_value=[VEC]) as model:
        embedding.embed_query("how to speak", client=r)
        embedding.embed_query("  how   to  speak ", client=r)
        assert model.call_count == 1
        embedding.embed_query("something else", client=r)
        assert model.call_count == 2


def test_key_depends_on_the_model(monkeypatch):
    a = embedding.cache_key("hello")
    monkeypatch.setattr(settings, "EMBED_MODEL", "some/other-model")
    assert embedding.cache_key("hello") != a  # vectors from different models must never be mixed


def test_wrong_sized_cache_entry_is_ignored():
    r = FakeRedis()
    r.data[embedding.cache_key("q")] = b"\x00" * 10
    with patch.object(embedding, "embed_batch", return_value=[VEC]) as model:
        assert embedding.embed_query("q", client=r) == VEC
    assert model.call_count == 1


def test_redis_down_never_breaks_search():
    class Broken:
        def get(self, key):
            raise redis.ConnectionError("down")

        def set(self, *a, **kw):
            raise redis.ConnectionError("down")

    with patch.object(embedding, "embed_batch", return_value=[VEC]):
        assert embedding.embed_query("q", client=Broken()) == VEC


def test_redis_write_failure_still_returns_vector():
    class WriteFails(FakeRedis):
        def set(self, *a, **kw):
            raise redis.TimeoutError("slow")

    with patch.object(embedding, "embed_batch", return_value=[VEC]):
        assert embedding.embed_query("q", client=WriteFails()) == VEC
