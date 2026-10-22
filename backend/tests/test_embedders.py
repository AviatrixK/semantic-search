"""Hosted embedding providers with a mocked HTTP transport: the exact request each one sends (task types, dimensions, auth),
batching, normalisation, token counting, retries/timeouts, and the query cache. No network, no keys, no model."""
import json
import math
from unittest.mock import patch

import httpx
import pytest
import redis

from app.core.config import settings
from app.services import embedders
from app.services.embedders import (CohereEmbedder, EmbeddingNotConfigured, EmbeddingRejected, EmbeddingUnavailable, GeminiEmbedder,
                                    LocalEmbedder, OpenAIEmbedder, cache_key, make_embedder, normalize, query_vector)

DIM = 8


class Recorder:
    """An httpx transport that records every request and answers from `respond(request, body)`."""

    def __init__(self, respond):
        self.respond, self.requests = respond, []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request, body))
        out = self.respond(request, body)
        if isinstance(out, Exception):
            raise out
        return out

    @property
    def bodies(self):
        return [b for _, b in self.requests]


def vec(n=DIM, v=3.0):
    return [v] * n  # not unit length on purpose: the embedder must normalise


def gemini_ok(request, body):
    return httpx.Response(200, json={"embedding": {"values": vec()}, "usageMetadata": {"promptTokenCount": 7}})


def cohere_ok(request, body):
    n = len(body["texts"])
    return httpx.Response(200, json={"embeddings": {"float": [vec() for _ in range(n)]}, "meta": {"billed_units": {"input_tokens": 5 * n}}})


def openai_ok(request, body):
    inputs = body["input"]
    data = [{"index": i, "embedding": [5.0 if j == i else 0.0 for j in range(DIM)]} for i in range(len(inputs))]  # one-hot: vector i points at axis i
    return httpx.Response(200, json={"data": list(reversed(data)), "usage": {"total_tokens": 4 * len(inputs)}})  # reversed on purpose


def build(cls, rec, model, **kw):
    return cls(model=model, dim=kw.pop("dim", DIM), api_key=kw.pop("api_key", "secret-key"), client=rec.client, sleep=lambda s: None)


def norm(v):
    return math.sqrt(sum(x * x for x in v))


# ---------------------------------------------------------------- Gemini
def test_gemini_embedding_2_puts_the_task_in_the_text_not_in_a_task_type_field():
    rec = Recorder(gemini_ok)
    e = build(GeminiEmbedder, rec, "gemini-embedding-2")
    q = e.embed_query("how do pauses help")
    d = e.embed_documents(["a spoken sentence"])[0]
    (rq, bq), (rd, bd) = rec.requests
    assert str(rq.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent"
    assert rq.headers["x-goog-api-key"] == "secret-key"
    assert bq["content"]["parts"][0]["text"] == "task: search result | query: how do pauses help"
    assert bd["content"]["parts"][0]["text"] == "title: none | text: a spoken sentence"
    assert "taskType" not in bq and "taskType" not in bd  # embedding-2 has no task_type
    assert bq["outputDimensionality"] == DIM == len(q) == len(d)
    assert norm(q) == pytest.approx(1.0)


def test_gemini_embedding_001_uses_task_types():
    rec = Recorder(gemini_ok)
    e = build(GeminiEmbedder, rec, "gemini-embedding-001")
    e.embed_query("q")
    e.embed_documents(["d"])
    bq, bd = rec.bodies
    assert (bq["taskType"], bd["taskType"]) == ("RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT")
    assert bq["content"]["parts"][0]["text"] == "q" and bd["content"]["parts"][0]["text"] == "d"  # no prefix for -001


def test_gemini_documents_keep_their_order_and_count_tokens():
    rec = Recorder(lambda r, b: httpx.Response(200, json={"embedding": {"values": [float(len(b["content"]["parts"][0]["text"]))] + [0.0] * (DIM - 1)}}))
    e = build(GeminiEmbedder, rec, "gemini-embedding-2")
    out = e.embed_documents(["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"])
    assert [round(v[0]) for v in out] == [1] * 7  # normalised: only the direction is kept, but one vector per text, in order
    assert len(rec.requests) == 7
    assert e.tokens_used >= 7  # no usageMetadata in this reply: estimated from the text


# ---------------------------------------------------------------- Cohere
def test_cohere_uses_search_query_and_search_document_and_output_dimension():
    rec = Recorder(cohere_ok)
    e = build(CohereEmbedder, rec, "embed-v5.0-pro")
    e.embed_query("q")
    e.embed_documents(["d1", "d2"])
    (rq, bq), (rd, bd) = rec.requests
    assert str(rq.url) == "https://api.cohere.com/v2/embed" and rq.headers["authorization"] == "Bearer secret-key"
    assert (bq["input_type"], bd["input_type"]) == ("search_query", "search_document")
    assert bq["embedding_types"] == ["float"] and bq["output_dimension"] == DIM and bq["model"] == "embed-v5.0-pro"
    assert e.tokens_used == 5 + 10  # billed_units.input_tokens


def test_cohere_v3_models_get_no_output_dimension():
    rec = Recorder(cohere_ok)
    build(CohereEmbedder, rec, "embed-multilingual-v3.0").embed_query("q")
    assert "output_dimension" not in rec.bodies[0]


def test_cohere_batches_of_at_most_96_texts():
    rec = Recorder(cohere_ok)
    out = build(CohereEmbedder, rec, "embed-v4.0").embed_documents([f"t{i}" for i in range(200)])
    assert len(out) == 200 and [len(b["texts"]) for b in rec.bodies] == [96, 96, 8]


# ---------------------------------------------------------------- OpenAI
def test_openai_sends_dimensions_and_restores_input_order():
    rec = Recorder(openai_ok)
    e = build(OpenAIEmbedder, rec, "text-embedding-3-large")
    out = e.embed_documents(["first", "second", "third"])
    (r, b), = rec.requests
    assert str(r.url) == "https://api.openai.com/v1/embeddings" and r.headers["authorization"] == "Bearer secret-key"
    assert b == {"model": "text-embedding-3-large", "input": ["first", "second", "third"], "dimensions": DIM, "encoding_format": "float"}
    assert all(norm(v) == pytest.approx(1.0) for v in out)
    assert e.tokens_used == 12
    assert [v.index(max(v)) for v in out] == [0, 1, 2]  # the reply came back reversed; each text still gets ITS vector


def test_openai_has_no_separate_query_type():
    rec = Recorder(openai_ok)
    e = build(OpenAIEmbedder, rec, "text-embedding-3-large")
    e.embed_query("same text")
    e.embed_documents(["same text"])
    q, d = rec.bodies
    assert q["input"] == d["input"] == ["same text"] and set(q) == set(d)  # symmetric model


def test_openai_batches_of_at_most_64():
    rec = Recorder(openai_ok)
    build(OpenAIEmbedder, rec, "text-embedding-3-small").embed_documents([f"t{i}" for i in range(130)])
    assert [len(b["input"]) for b in rec.bodies] == [64, 64, 2]


# ---------------------------------------------------------------- failures and retries
def test_a_503_is_retried_and_then_succeeds():
    calls = []

    def respond(request, body):
        calls.append(1)
        return httpx.Response(503) if len(calls) == 1 else gemini_ok(request, body)

    e = build(GeminiEmbedder, Recorder(respond), "gemini-embedding-2")
    assert len(e.embed_query("q")) == DIM and len(calls) == 2


def test_the_query_gets_exactly_one_retry_then_unavailable():
    rec = Recorder(lambda r, b: httpx.Response(500))
    with pytest.raises(EmbeddingUnavailable):
        build(OpenAIEmbedder, rec, "text-embedding-3-large").embed_query("q")
    assert len(rec.requests) == 1 + settings.EMBED_QUERY_RETRIES == 2


def test_timeouts_and_network_errors_become_unavailable():
    for exc in (httpx.ReadTimeout("slow"), httpx.ConnectError("refused")):
        rec = Recorder(lambda r, b, exc=exc: exc)
        with pytest.raises(EmbeddingUnavailable):
            build(CohereEmbedder, rec, "embed-v5.0-pro").embed_query("q")
        assert len(rec.requests) == 2


def test_rate_limit_429_counts_as_unavailable_not_rejected():
    rec = Recorder(lambda r, b: httpx.Response(429))
    with pytest.raises(EmbeddingUnavailable):
        build(GeminiEmbedder, rec, "gemini-embedding-2").embed_query("q")


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_a_client_error_is_rejected_at_once_without_retrying(code):
    rec = Recorder(lambda r, b: httpx.Response(code, json={"error": "secret-key leaked here?"}))
    with pytest.raises(EmbeddingRejected) as e:
        build(OpenAIEmbedder, rec, "text-embedding-3-large").embed_query("q")
    assert len(rec.requests) == 1 and "secret-key" not in str(e.value)


def test_a_missing_key_fails_before_any_request():
    rec = Recorder(gemini_ok)
    with pytest.raises(EmbeddingNotConfigured, match="GEMINI_API_KEY"):
        build(GeminiEmbedder, rec, "gemini-embedding-2", api_key="").embed_query("q")
    assert rec.requests == []


def test_queries_use_the_short_timeout_and_documents_the_long_one():
    rec = Recorder(gemini_ok)
    e = build(GeminiEmbedder, rec, "gemini-embedding-2")
    e.embed_query("q")
    e.embed_documents(["d"])
    q_timeout = rec.requests[0][0].extensions["timeout"]["read"]
    d_timeout = rec.requests[1][0].extensions["timeout"]["read"]
    assert q_timeout == settings.EMBED_QUERY_TIMEOUT_SEC == 2.0
    assert d_timeout == settings.EMBED_DOC_TIMEOUT_SEC


def test_an_unexpected_reply_shape_is_rejected_not_crashed():
    rec = Recorder(lambda r, b: httpx.Response(200, json={"surprise": True}))
    for cls, model in ((GeminiEmbedder, "gemini-embedding-2"), (CohereEmbedder, "embed-v5.0-pro"), (OpenAIEmbedder, "text-embedding-3-large")):
        with pytest.raises(EmbeddingRejected):
            build(cls, rec, model).embed_query("q")


# ---------------------------------------------------------------- factory
def test_make_embedder_uses_settings_and_validates_the_provider():
    e = make_embedder("gemini")
    assert (e.provider, e.model, e.dim, e.key) == ("gemini", settings.GEMINI_EMBED_MODEL, settings.EMBED_DIMENSIONS,
                                                 f"gemini/{settings.GEMINI_EMBED_MODEL}@{settings.EMBED_DIMENSIONS}")
    assert make_embedder("cohere", model="embed-v4.0", dim=1024).key == "cohere/embed-v4.0@1024"
    local = make_embedder("local")
    assert isinstance(local, LocalEmbedder) and local.uses_legacy_column and local.dim == settings.EMBED_DIM
    assert not e.uses_legacy_column
    with pytest.raises(ValueError, match="unknown embedding provider"):
        make_embedder("llama")


def test_get_embedder_follows_embed_provider():
    with patch.object(settings, "EMBED_PROVIDER", "openai"):
        assert embedders.get_embedder().provider == "openai"
    assert embedders.get_embedder().provider == "local"  # the default stays local


def test_normalize():
    assert norm(normalize([3.0, 4.0])) == pytest.approx(1.0)
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


# ---------------------------------------------------------------- query cache and keys
class FakeRedis:
    def __init__(self):
        self.data, self.ttl = {}, {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key], self.ttl[key] = value, ex


def test_cache_key_is_provider_model_dimensions_and_normalised_query():
    a = GeminiEmbedder(model="m1", dim=8, api_key="k")
    assert cache_key(a, "how to  speak ") == cache_key(a, "how to speak")
    assert len({cache_key(a, "q"), cache_key(GeminiEmbedder(model="m2", dim=8, api_key="k"), "q"),
                cache_key(GeminiEmbedder(model="m1", dim=16, api_key="k"), "q"), cache_key(OpenAIEmbedder(model="m1", dim=8, api_key="k"), "q")}) == 4
    assert cache_key(a, "q").startswith("emb:") and len(cache_key(a, "q")) == 4 + 64


def test_query_vector_caches_for_seven_days_and_the_second_call_makes_no_request():
    rec, r = Recorder(gemini_ok), FakeRedis()
    e = build(GeminiEmbedder, rec, "gemini-embedding-2")
    first = query_vector(e, "robots", client=r)
    second = query_vector(e, "  robots ", client=r)
    assert second == pytest.approx(first, abs=1e-6) and len(rec.requests) == 1
    (key,) = r.data
    assert r.ttl[key] == 7 * 86400 == settings.EMBED_CACHE_TTL_SEC


def test_a_wrong_sized_cache_entry_is_ignored():
    rec, r = Recorder(gemini_ok), FakeRedis()
    e = build(GeminiEmbedder, rec, "gemini-embedding-2")
    r.data[cache_key(e, "q")] = b"\x00" * 7
    assert len(query_vector(e, "q", client=r)) == DIM and len(rec.requests) == 1


def test_redis_down_never_breaks_a_query():
    class Broken:
        def get(self, key):
            raise redis.ConnectionError("down")

        def set(self, *a, **k):
            raise redis.ConnectionError("down")

    rec = Recorder(gemini_ok)
    assert len(query_vector(build(GeminiEmbedder, rec, "gemini-embedding-2"), "q", client=Broken())) == DIM


def test_an_unavailable_provider_is_not_cached_and_raises():
    rec, r = Recorder(lambda req, b: httpx.Response(503)), FakeRedis()
    with pytest.raises(EmbeddingUnavailable):
        query_vector(build(GeminiEmbedder, rec, "gemini-embedding-2"), "q", client=r)
    assert r.data == {}


# ---------------------------------------------------------------- token counts: the provider's own number wins, an estimate is the fallback
def test_gemini_uses_the_reported_prompt_token_count_and_estimates_nothing():
    rec = Recorder(lambda r, b: httpx.Response(200, json={"embedding": {"values": vec()}, "usageMetadata": {"promptTokenCount": 41}}))
    e = build(GeminiEmbedder, rec, "gemini-embedding-2")
    e.embed_query("a text that would be estimated as only a few tokens")
    assert (e.tokens_used, e.tokens_estimated) == (41, 0)  # not len(text) // 4


def test_gemini_estimates_only_when_the_reply_has_no_usage_and_says_how_much():
    rec = Recorder(lambda r, b: httpx.Response(200, json={"embedding": {"values": vec()}}))
    e = build(GeminiEmbedder, rec, "gemini-embedding-2")
    e.embed_query("x" * 400)
    assert e.tokens_used == e.tokens_estimated == embedders.estimate_tokens("x" * 400) == 100


def test_a_provider_can_mix_reported_and_estimated_counts_and_they_are_kept_apart():
    replies = iter([{"usageMetadata": {"promptTokenCount": 10}}, {}, {"usageMetadata": {"promptTokenCount": 7}}])
    rec = Recorder(lambda r, b: httpx.Response(200, json={"embedding": {"values": vec()}, **next(replies)}))
    e = build(GeminiEmbedder, rec, "gemini-embedding-2")
    for t in ("one", "y" * 80, "three"):
        e.embed_query(t)
    assert e.tokens_estimated == embedders.estimate_tokens("y" * 80) == 20
    assert e.tokens_used == 10 + 20 + 7


def test_cohere_and_openai_report_their_own_counts_and_a_missing_or_odd_field_falls_back_to_the_estimate():
    rec = Recorder(cohere_ok)
    c = build(CohereEmbedder, rec, "embed-v5.0-pro")
    c.embed_documents(["aa", "bb"])
    assert (c.tokens_used, c.tokens_estimated) == (10, 0)
    rec = Recorder(lambda r, b: httpx.Response(200, json={"embeddings": {"float": [vec()]}}))
    c = build(CohereEmbedder, rec, "embed-v5.0-pro")
    c.embed_documents(["z" * 40])
    assert (c.tokens_used, c.tokens_estimated) == (10, 10)
    for bad in ("12", None, True, 3.5):  # only a real integer counts as a reported figure (a bool is an int in Python: not a count)
        rec = Recorder(lambda r, b, bad=bad: httpx.Response(200, json={"data": [{"index": 0, "embedding": vec()}], "usage": {"total_tokens": bad}}))
        o = build(OpenAIEmbedder, rec, "text-embedding-3-large")
        o.embed_documents(["w" * 40])
        assert (o.tokens_used, o.tokens_estimated) == (10, 10), bad
    rec = Recorder(openai_ok)
    o = build(OpenAIEmbedder, rec, "text-embedding-3-large")
    o.embed_documents(["a", "b", "c"])
    assert (o.tokens_used, o.tokens_estimated) == (12, 0)


def test_the_local_model_has_no_token_counts():
    assert (LocalEmbedder.tokens_used, LocalEmbedder.tokens_estimated) == (0, 0)
