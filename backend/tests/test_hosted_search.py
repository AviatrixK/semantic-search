"""Search with a HOSTED embedding provider on the real test database: vectors from chunk_embeddings, keyword search across the native,
romanised and English text, the keyword-only fallback when the provider fails, and model-free (lexical) highlights.
The provider is a fake: no network, no keys, and the local model is never loaded."""
import logging
import math
import uuid
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy import text

from app.core.config import settings
from app.models import Chunk, Video
from app.services import embedders, retrieval, vectorstore
from app.services.embedders import EmbeddingUnavailable, GeminiEmbedder


def unit(*xs):
    n = math.sqrt(sum(x * x for x in xs))
    return [x / n for x in xs]


E0, E1, E2, E3 = unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0), unit(0, 0, 0, 1)


class FakeHosted(GeminiEmbedder):
    """A hosted provider with scripted query vectors (no HTTP). `fail` makes it behave like an outage."""

    def __init__(self, vectors=None, fail=None):
        super().__init__(model="fake-emb", dim=4, api_key="k")
        self.vectors, self.fail, self.queries = vectors or {}, fail, []

    def embed_query(self, text):
        self.queries.append(text)
        if self.fail:
            raise self.fail
        return self.vectors.get(text, E3)


@pytest.fixture
def lib(api, db_session):
    """A Hindi video (native + roman + English text) and an English video with multi-sentence chunks."""
    hindi = Video(id=uuid.uuid4(), title="Hindi talk", storage_key="raw/h.mp4")
    eng = Video(id=uuid.uuid4(), title="English talk", storage_key="raw/e.mp4")
    db_session.add_all([hindi, eng])
    db_session.flush()
    c = lambda v, i, native, roman=None, en=None: Chunk(id=uuid.uuid4(), video_id=v.id, idx=i, start_sec=i * 30.0, end_sec=i * 30.0 + 30,
                                                        text=native, text_native=native, text_roman=roman, text_en=en)
    c0 = c(hindi, 0, "मेरा फोन नंबर है नौ आठ", "mera phone number hai nau aath", "my phone number is nine eight")
    c1 = c(hindi, 1, "हम आज सपने के बारे में बात करेंगे", "hum aaj sapne ke baare mein baat karenge", "we will talk about dreams today")
    c2 = c(eng, 0, "The weather is nice. The report for tomorrow says rain. Bring an umbrella.")
    db_session.add_all([c0, c1, c2])
    db_session.commit()
    emb = FakeHosted({"phone": E1, "phone number": E1, "dreams": E2, "weather": E3})
    vectorstore.store(db_session, emb, [
        (c0.id, "native", E0), (c0.id, "en", E1),  # the English variant is what matches the query "phone"
        (c1.id, "native", unit(0, 0, 1, 0.2)), (c1.id, "en", E2),
        (c2.id, "native", E3)])
    db_session.commit()
    api.post("/auth/register", json={"email": "ann@example.com", "password": "Passw0rd123"})
    token = api.post("/auth/login", json={"email": "ann@example.com", "password": "Passw0rd123"}).json()["access_token"]
    api.headers.update({"Authorization": f"Bearer {token}"})
    with patch.object(embedders, "get_embedder", return_value=emb), \
         patch.object(retrieval, "embed_batch", side_effect=AssertionError("the local model must not run for a hosted provider")), \
         patch.object(retrieval, "embed_query", side_effect=AssertionError("the local model must not run for a hosted provider")):
        api.c0, api.c1, api.c2, api.emb, api.db = c0, c1, c2, emb, db_session
        yield api


def titles(hits):
    return [(h["title"], h["start_sec"]) for h in hits]


# ---------------------------------------------------------------- vectors from chunk_embeddings
def test_vector_mode_ranks_by_the_hosted_models_vectors_and_takes_the_better_text_variant(lib):
    hits = lib.get("/api/search?q=phone&mode=vector").json()
    assert titles(hits)[0] == ("Hindi talk", 0)
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-3)  # matched through the English vector, not the poor native one
    assert lib.emb.queries == ["phone"]


def test_search_finds_a_hindi_chunk_with_an_english_query_through_its_english_vector(lib):
    hits = lib.get("/api/search?q=dreams&mode=vector").json()
    assert titles(hits)[0] == ("Hindi talk", 30)


def test_hybrid_merges_the_keyword_and_vector_sides_for_a_hosted_model(lib):
    top = lib.get("/api/search?q=phone+number").json()[0]
    assert (top["title"], top["start_sec"]) == ("Hindi talk", 0)
    assert set(top["found_by"]) == {"keyword", "vector"} and top["rrf"] > 0


def test_a_keyword_only_hit_still_gets_the_hosted_cosine_as_its_score(lib):
    """min_score above the real cosine (0.995) drops the chunk from the vector side, so it arrives through the keyword side only:
    its score must still be the cosine with this model's vector (vectorstore.score_chunks), not a rank-based number."""
    lib.emb.vectors["nine"] = unit(0, 1, 0.1, 0)
    hits = retrieval.hybrid_search(lib.db, "nine", k=5, min_score=0.9999, highlight=False, embedder=lib.emb)
    (hit,) = [h for h in hits if h["chunk_id"] == lib.c0.id]
    assert hit["found_by"] == ["keyword"]
    assert hit["score"] == pytest.approx(unit(0, 1, 0.1, 0)[1], abs=1e-3)


# ---------------------------------------------------------------- keyword search: two indexes ('simple' over native + romanised, 'english' over English)
@pytest.mark.parametrize("query,expected", [
    ("फोन नंबर", ("Hindi talk", 0)),        # native script
    ("mera phone", ("Hindi talk", 0)),       # romanised: the Hindi chunk is found by typing Hindi in Latin letters
    ("dreams today", ("Hindi talk", 30)),    # English translation
    ("umbrella", ("English talk", 0)),       # a chunk with no variants at all still works (falls back to `text`)
])
def test_keyword_search_matches_native_romanised_and_english_text(lib, query, expected):
    assert expected in titles(lib.get("/api/search", params={"q": query, "mode": "keyword"}).json())
    assert lib.emb.queries == []  # keyword mode never calls the embedding provider


def test_the_two_generated_columns_split_the_variants_between_them(lib):
    tsv, tsv_en = (lib.db.execute(text(f"SELECT {col}::text FROM chunks WHERE id = :i"), {"i": lib.c0.id}).scalar() for col in ("tsv", "tsv_en"))
    assert "mera" in tsv and "phone" in tsv and "फोन" in tsv          # 'simple': native script and romanisation, as written
    assert "nine" not in tsv                                             # the English text is NOT in the 'simple' index any more
    assert "'phone'" in tsv_en and "'nine'" in tsv_en and "mera" not in tsv_en and "फोन" not in tsv_en  # 'english': the English text only
    assert "'number'" in tsv_en and "'is'" not in tsv_en                 # ... with stop words removed
    umbrella = lib.db.execute(text("SELECT tsv::text, tsv_en::text FROM chunks WHERE id = :i"), {"i": lib.c2.id}).one()
    assert "umbrella" in umbrella[0] and "umbrella" in umbrella[1]       # no variants (Whisper English): `text` feeds both


def test_english_text_is_stemmed_but_romanised_and_native_text_are_not(lib):
    # c1's English is "we will talk about dreams today": the stemmed index finds it from the singular, and from another form of talk
    assert ("Hindi talk", 30) in titles(lib.get("/api/search", params={"q": "dream", "mode": "keyword"}).json())
    assert ("Hindi talk", 30) in titles(lib.get("/api/search", params={"q": "talking", "mode": "keyword"}).json())
    # the romanisation "baat karenge" is matched exactly as written (no stemming for a language Postgres has no stemmer for)
    assert ("Hindi talk", 30) in titles(lib.get("/api/search", params={"q": "karenge", "mode": "keyword"}).json())
    assert lib.get("/api/search", params={"q": "karen", "mode": "keyword"}).json() == []


def test_a_chunk_found_by_both_indexes_ranks_above_one_found_by_one(lib):
    db = lib.db
    both = Chunk(id=uuid.uuid4(), video_id=lib.c0.video_id, idx=5, start_sec=500, end_sec=530, text="telephone telephone", text_native="telephone telephone",
                 text_roman="telephone", text_en="telephone")
    only_simple = Chunk(id=uuid.uuid4(), video_id=lib.c0.video_id, idx=6, start_sec=600, end_sec=630, text="telephone", text_native="telephone", text_en="y")
    only_english = Chunk(id=uuid.uuid4(), video_id=lib.c0.video_id, idx=7, start_sec=700, end_sec=730, text="x", text_native="x", text_en="telephone")
    db.add_all([both, only_simple, only_english])
    db.commit()
    hits = retrieval.keyword_search(db, "telephone", k=10, highlight=False, embedder=lib.emb)
    order = [h["start_sec"] for h in hits]
    assert order[0] == 500 and set(order) == {500, 600, 700}  # found by 'simple' AND 'english' first; each single-index match still found


# ---------------------------------------------------------------- resilience: provider down -> keyword-only
def test_hybrid_falls_back_to_keyword_only_when_the_embedding_fails_and_says_so(lib, caplog):
    lib.emb.fail = EmbeddingUnavailable("gemini unavailable: ReadTimeout")
    with caplog.at_level(logging.WARNING):
        r = lib.get("/api/search?q=phone+number")
    assert r.status_code == 200
    hits = r.json()
    assert titles(hits) == [("Hindi talk", 0)] and hits[0]["found_by"] == ["keyword"]
    assert r.headers["X-Search-Fallback"] == "keyword-only"
    warn = [x for x in caplog.records if x.levelno == logging.WARNING and "falling back to keyword-only" in x.getMessage()]
    assert warn and "EmbeddingUnavailable" in warn[0].getMessage()


def test_vector_mode_also_falls_back_to_keywords(lib):
    lib.emb.fail = EmbeddingUnavailable("down")
    r = lib.get("/api/search?q=dreams&mode=vector")
    assert titles(r.json()) == [("Hindi talk", 30)] and r.headers["X-Search-Fallback"] == "keyword-only"


def test_no_fallback_header_when_everything_works(lib):
    assert "X-Search-Fallback" not in lib.get("/api/search?q=phone").headers


def test_a_missing_api_key_also_degrades_to_keywords_instead_of_failing_the_search(lib):
    lib.emb.fail = embedders.EmbeddingNotConfigured("GEMINI_API_KEY is empty")
    assert lib.get("/api/search?q=umbrella").status_code == 200


def test_a_real_provider_that_times_out_costs_one_retry_then_search_continues(api, db_session, lib, caplog):
    """The whole chain with the real GeminiEmbedder over a failing transport: 1 try + 1 retry, then keyword-only."""
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("slow")

    real = GeminiEmbedder(model="fake-emb", dim=4, api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    with patch.object(embedders, "get_embedder", return_value=real), caplog.at_level(logging.WARNING):
        r = api.get("/api/search?q=phone+number")
    assert r.status_code == 200 and titles(r.json()) == [("Hindi talk", 0)]
    assert len(calls) == 2  # EMBED_QUERY_RETRIES = 1
    assert r.headers["X-Search-Fallback"] == "keyword-only"


def test_a_failed_embedding_is_not_cached_so_the_next_search_tries_again(lib):
    lib.emb.fail = EmbeddingUnavailable("down")
    lib.get("/api/search?q=phone")
    lib.emb.fail = None
    r = lib.get("/api/search?q=phone&mode=vector")
    assert "X-Search-Fallback" not in r.headers and titles(r.json())[0] == ("Hindi talk", 0)


# ---------------------------------------------------------------- highlights without a model
def test_hosted_providers_get_lexical_highlights_and_no_sentence_embedding(lib):
    r = lib.get("/api/search?q=report+tomorrow&mode=keyword")
    (hit,) = r.json()
    assert hit["highlight"] == "The report for tomorrow says rain."
    assert "lexical=1" in r.headers["X-Search-Highlight"]  # (embed_batch / embed_query would have raised: see the fixture)


def test_a_chunk_with_no_shared_word_is_shown_without_a_highlight_not_a_wrong_one(lib):
    hits = lib.get("/api/search?q=weather&mode=vector").json()
    assert hits[0]["title"] == "English talk" and hits[0]["highlight"] == "The weather is nice."  # shares "weather"
    unrelated = lib.get("/api/search?q=nonexistentword&mode=vector").json()[0]  # its vector lands on the English chunk, but no word matches
    assert unrelated["title"] == "English talk" and unrelated["highlight"] is None


def test_a_single_sentence_chunk_highlights_itself(lib):
    hindi = lib.get("/api/search?q=dreams&mode=vector").json()[0]
    assert hindi["title"] == "Hindi talk" and hindi["highlight"] == hindi["text"]


def test_highlights_stay_off_when_asked(lib):
    hits = lib.get("/api/search?q=umbrella&mode=keyword&highlight=false").json()
    assert hits and all(h["highlight"] is None for h in hits)


def test_the_fallback_still_highlights_without_the_provider(lib):
    lib.emb.fail = EmbeddingUnavailable("down")
    (hit,) = lib.get("/api/search?q=umbrella").json()
    assert hit["highlight"] == "Bring an umbrella."


# ---------------------------------------------------------------- MIN_SCORE per provider
def test_each_provider_has_its_own_cut_off_and_inherits_min_score_when_it_has_none():
    with patch.object(settings, "MIN_SCORE", 0.25), patch.object(settings, "MIN_SCORE_LOCAL", None), patch.object(settings, "MIN_SCORE_GEMINI", 0.6), \
         patch.object(settings, "MIN_SCORE_COHERE", None), patch.object(settings, "MIN_SCORE_OPENAI", 0.0):
        assert settings.min_score_for("local") == 0.25 and settings.min_score_for("cohere") == 0.25  # unset: the general value
        assert settings.min_score_for("gemini") == 0.6
        assert settings.min_score_for("openai") == 0.0  # an explicit 0 is a value, not "unset"
        assert settings.min_score_for("unknown-provider") == 0.25


def test_blank_provider_cut_offs_in_the_environment_mean_unset():
    from app.core.config import Settings
    s = Settings(MIN_SCORE_GEMINI="", MIN_SCORE_COHERE=" ", MIN_SCORE_OPENAI="0.4", _env_file=None)
    assert (s.MIN_SCORE_GEMINI, s.MIN_SCORE_COHERE, s.MIN_SCORE_OPENAI) == (None, None, 0.4)


def test_vector_search_applies_the_cut_off_of_the_active_provider(lib):
    # the query "phone" matches c0 with cosine 1.0 through its English vector (see the first test of this file)
    assert lib.emb.provider == "gemini"
    with patch.object(settings, "MIN_SCORE_GEMINI", 0.5):
        assert titles(retrieval.vector_search(lib.db, "phone", k=5, highlight=False, embedder=lib.emb))[:1] == [("Hindi talk", 0)]
    with patch.object(settings, "MIN_SCORE_GEMINI", 1.01):
        assert retrieval.vector_search(lib.db, "phone", k=5, highlight=False, embedder=lib.emb) == []  # nothing reaches 1.01
    with patch.object(settings, "MIN_SCORE_GEMINI", 1.01), patch.object(settings, "MIN_SCORE_COHERE", 0.0), patch.object(settings, "MIN_SCORE", 0.0):
        assert retrieval.vector_search(lib.db, "phone", k=5, highlight=False, embedder=lib.emb) == []  # another provider's value is not used
    with patch.object(settings, "MIN_SCORE_GEMINI", 1.01):  # an explicit argument still wins (the eval harness uses it)
        assert retrieval.vector_search(lib.db, "phone", k=5, highlight=False, embedder=lib.emb, min_score=0.0)


def test_hybrid_search_uses_the_same_per_provider_cut_off_for_its_vector_side(lib):
    with patch.object(settings, "MIN_SCORE_GEMINI", 1.01):
        hits = retrieval.hybrid_search(lib.db, "phone", k=5, highlight=False, embedder=lib.emb)
    assert all("vector" not in h["found_by"] for h in hits)  # the vector side was cut off entirely; only keyword hits can remain
    with patch.object(settings, "MIN_SCORE_GEMINI", 0.5):
        hits = retrieval.hybrid_search(lib.db, "phone", k=5, highlight=False, embedder=lib.emb)
    assert any("vector" in h["found_by"] for h in hits)


def test_every_provider_cut_off_is_documented_in_the_env_examples():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for name in (".env.example", ".env.prod.example"):
        path = root / name
        if not path.exists():
            pytest.skip(f"{name} is not part of this container")
        text_ = path.read_text(encoding="utf-8")
        for var in ("MIN_SCORE", "MIN_SCORE_LOCAL", "MIN_SCORE_GEMINI", "MIN_SCORE_COHERE", "MIN_SCORE_OPENAI"):
            assert var in text_, f"{var} missing in {name}"
