"""The provider comparison: cost arithmetic checked by hand, and a full run on the real test database with a good, a poor, an empty, an
unconfigured and a failing provider. No network: the providers are fakes."""
import json
import math
import uuid
from unittest.mock import patch

import pytest

from app.models import Chunk, Video
from app.services import embedders, pricing, vectorstore
from app.services.embedders import EmbeddingUnavailable, GeminiEmbedder
from eval import costs, metrics, run_provider_eval as rpe


def unit(*xs):
    n = math.sqrt(sum(x * x for x in xs))
    return [x / n for x in xs]


# ---------------------------------------------------------------- cost arithmetic
def corpus(hours=2.0, native=80_000, en=80_000, chunks=100, videos=2):
    return costs.Corpus(videos=videos, hours=hours, native_chars=native, en_chars=en, chunks=chunks)


def test_average_query_tokens_uses_four_characters_per_token():
    assert costs.average_query_tokens(["a" * 12, "b" * 20, "c" * 4]) == pytest.approx((3 + 5 + 1) / 3)
    assert costs.average_query_tokens([]) == 0.0


def test_cost_per_1000_searches_is_tokens_times_price():
    price = pricing.EMBEDDING_PRICES["gemini/gemini-embedding-2"]  # $0.20 per 1M tokens
    assert costs.per_1000_searches_usd(10, price) == pytest.approx(10 * 1000 / 1e6 * 0.20)  # $0.002


def test_cost_per_video_hour_counts_every_embedded_text_variant():
    price = pricing.EMBEDDING_PRICES["openai/text-embedding-3-large"]  # $0.13 per 1M tokens
    c = corpus(hours=2.0, native=80_000, en=80_000)  # 160,000 chars = 40,000 tokens over 2 h = 20,000 tokens per hour
    assert costs.tokens_per_hour(c) == pytest.approx(20_000)
    assert costs.per_video_hour_usd(c, price) == pytest.approx(20_000 / 1e6 * 0.13)


def test_a_whisper_era_corpus_can_be_costed_as_if_it_had_english_variants():
    price = pricing.EMBEDDING_PRICES["cohere/embed-v5.0-pro"]
    whisper_only = corpus(native=80_000, en=0)
    one = costs.per_video_hour_usd(whisper_only, price)
    both = costs.per_video_hour_usd(whisper_only, price, assume_both_variants=True)
    assert both == pytest.approx(2 * one)
    assert costs.per_video_hour_usd(corpus(native=80_000, en=80_000), price, assume_both_variants=True) == pytest.approx(both)  # already has en


def test_an_empty_or_zero_hour_corpus_never_divides_by_zero():
    price = pricing.EMBEDDING_PRICES["gemini/gemini-embedding-2"]
    assert costs.per_video_hour_usd(corpus(hours=0.0), price) is None
    assert costs.corpus_tokens(corpus(native=0, en=0)) == 0


def test_unknown_prices_are_reported_as_unknown_never_guessed():
    row = costs.cost_row("cohere/embed-v4.0@1536", 10, corpus())  # no official per-token price was found for v4
    assert row.price is None and row.per_1000_searches_usd is None and row.per_video_hour_usd is None
    assert costs.fmt_usd(None) == "unknown" and costs.fmt_usd(0.0) == "free" and costs.fmt_usd(0.002) == "$0.0020" and costs.fmt_usd(12.5) == "$12.50"


def test_the_local_model_is_free_and_the_key_ignores_the_dimension_suffix():
    row = costs.cost_row("local/sentence-transformers/all-MiniLM-L6-v2@384", 10, corpus())
    assert row.per_1000_searches_usd == 0 and row.per_video_hour_usd == 0
    assert pricing.embedding_price("gemini/gemini-embedding-2@768") == pricing.embedding_price("gemini/gemini-embedding-2@3072")


def test_sarvam_is_one_transcription_job_plus_the_text_apis_and_whisper_costs_nothing():
    stt = pricing.SARVAM_STT_INR_PER_HOUR
    assert stt == 30 and pricing.SARVAM_TEXT_INR_PER_10K_CHARS == 20
    # per hour of video: 30 (ONE job, whatever the number of text versions) + 20 per 10K characters of each version
    assert costs.transcription_per_hour_inr("sarvam", 0) == 30
    assert costs.transcription_per_hour_inr("sarvam", 1, 50_000) == pytest.approx(30 + 50_000 / 10_000 * 20)
    assert costs.transcription_per_hour_inr("sarvam", 2, 50_000) == pytest.approx(30 + 2 * 50_000 / 10_000 * 20)
    assert costs.transcription_per_hour_inr("sarvam", 2) == pytest.approx(30 + 2 * pricing.TEXT_CHARS_PER_AUDIO_HOUR / 10_000 * 20)  # measured default
    assert costs.transcription_per_hour_inr("whisper") == 0
    assert pricing.sarvam_cost_inr(5400, variants=2, text_chars=100_000) == pytest.approx(1.5 * 30 + 2 * 100_000 / 10_000 * 20)  # 90 minutes
    assert pricing.sarvam_cost_inr(3600, variants=0) == 30  # no text versions: the audio only


def test_the_measured_text_volume_is_what_the_default_estimate_uses():
    assert pricing.TEXT_CHARS_PER_AUDIO_HOUR == 78_000  # 169,177 characters of chunk text / 2.17 h on the dev library
    assert pricing.sarvam_cost_inr(3600) == pytest.approx(30 + 2 * 78_000 / 10_000 * 20) == pytest.approx(342.0)


def test_the_combined_total_needs_an_exchange_rate_for_sarvam_and_never_assumes_one():
    row = costs.cost_row("gemini/gemini-embedding-2@1536", 10, corpus())
    inr = costs.transcription_per_hour_inr("sarvam", 2)
    assert costs.combined_per_hour_usd(row, "sarvam", 2, None) is None
    assert costs.combined_per_hour_usd(row, "sarvam", 2, 90.0) == pytest.approx(row.per_video_hour_usd + inr / 90.0)
    assert costs.combined_per_hour_usd(row, "whisper", 2, None) == pytest.approx(row.per_video_hour_usd)  # no INR involved


def test_every_price_has_a_source_link_and_unconfirmed_ones_say_so():
    for key, p in pricing.EMBEDDING_PRICES.items():
        assert p.source.startswith("https://") and p.usd_per_million_tokens > 0, key
    assert pricing.EMBEDDING_PRICES["gemini/gemini-embedding-001"].confirmed is False  # only in a 2025 post, not on the pricing page
    assert pricing.SARVAM_SOURCE.startswith("https://docs.sarvam.ai")


# ---------------------------------------------------------------- a whole run on the database
class FakeHosted(GeminiEmbedder):
    """Query vectors scripted per query text; `fail` makes every call raise (an outage)."""

    def __init__(self, name, vectors, key="k", fail=None, tokens_per_query=0):
        super().__init__(model=name, dim=4, api_key=key)
        self.vectors, self.fail, self.queries, self.per_query = vectors, fail, [], tokens_per_query

    def embed_query(self, text):
        self.queries.append(text)
        if self.fail:
            raise self.fail
        self.tokens_used += self.per_query
        return self.vectors[text]


QUERIES = {"how do pauses help": (0, unit(1, 0, 0, 0)), "what is said about dreams": (1, unit(0, 1, 0, 0)), "who is the speaker": (2, unit(0, 0, 1, 0))}


@pytest.fixture
def world(api, db_session, tmp_path):
    v = Video(id=uuid.uuid4(), title="Talk", storage_key="raw/t.mp4", duration_sec=3600)
    db_session.add(v)
    db_session.flush()
    chunks = [Chunk(id=uuid.uuid4(), video_id=v.id, idx=i, start_sec=i * 30.0, end_sec=i * 30.0 + 30, text=f"text number {i} pauses dreams speaker",
                    text_native=f"text number {i}", text_en=f"english {i}") for i in range(3)]
    db_session.add_all(chunks)
    db_session.commit()
    gold = tmp_path / "queries.jsonl"
    gold.write_text("\n".join(json.dumps({"query": q, "video_id": str(v.id), "start_sec": idx * 30.0 + 1, "end_sec": idx * 30.0 + 29, "type": "paraphrase"})
                              for q, (idx, _) in QUERIES.items()), encoding="utf-8")
    good = FakeHosted("good", {q: vec for q, (_, vec) in QUERIES.items()}, tokens_per_query=7)
    wrong = {q: unit(*[1.0 if j == (idx + 1) % 3 else 0.0 for j in range(4)]) for q, (idx, _) in QUERIES.items()}  # each query -> the NEXT chunk
    poor = FakeHosted("poor", wrong)
    for emb in (good, poor):  # both models have vectors for every chunk: chunk i points along axis i
        vectorstore.store(db_session, emb, [(c.id, "native", unit(*[1.0 if j == i else 0.0 for j in range(4)])) for i, c in enumerate(chunks)])
    db_session.commit()
    cache = rpe.MemoryCache()
    with patch.object(embedders, "_redis", return_value=cache):
        yield type("W", (), {"db": db_session, "video": v, "gold": gold, "good": good, "poor": poor, "cache": cache,
                             "queries": metrics.load_queries(gold), "ids": [v.id]})


def evaluate(w, embs, **kw):
    c = costs.Corpus(videos=1, hours=1.0, native_chars=4000, en_chars=1000, chunks=3)
    return rpe.run_provider_eval(w.db, w.queries, embs, k=5, repeats=2, cache=w.cache, corpus=c, video_ids=w.ids, progress=lambda m: None, **kw)


def test_a_good_provider_scores_higher_than_a_poor_one_on_the_same_chunks(world):
    out = evaluate(world, [world.good, world.poor])["providers"]
    g, p = out[world.good.key]["modes"]["vector"]["summary"], out[world.poor.key]["modes"]["vector"]["summary"]
    assert (g["recall@1"], g["recall@5"], g["mrr"]) == (1.0, 1.0, 1.0)  # every query lands on its gold chunk first
    assert (p["recall@1"], p["recall@5"], p["mrr"]) == (0.0, 0.0, 0.0)  # its neighbours are the wrong chunks (and too dissimilar to pass MIN_SCORE)
    assert set(out[world.good.key]["modes"]) == {"vector", "hybrid"}
    assert g["n"] == 3 and g["median_latency_ms"] is not None


def test_each_query_is_embedded_exactly_once_per_provider_because_repeats_are_cached(world):
    evaluate(world, [world.good])
    assert sorted(world.good.queries) == sorted(QUERIES)  # 3 queries, 3 API calls: vector + hybrid + 2 repeats all hit the cache


def test_cold_latency_is_measured_on_the_first_uncached_search(world):
    ticks = iter(range(0, 10_000))
    clock = lambda: next(ticks) / 1000.0  # every clock() call advances 1 ms
    out = evaluate(world, [world.good], clock=clock)["providers"][world.good.key]
    assert out["cold_median_ms"] == pytest.approx(1.0)  # one start/stop pair around each cold search


def test_query_tokens_come_from_the_provider_when_it_reports_them_else_they_are_estimated(world):
    out = evaluate(world, [world.good, world.poor])["providers"]
    assert out[world.good.key]["query_tokens"] == {"avg": 7.0, "source": "reported by the provider"}
    assert out[world.poor.key]["query_tokens"]["source"].startswith("estimated")


def test_costs_are_attached_and_unknown_models_say_unknown(world):
    out = evaluate(world, [world.good])["providers"][world.good.key]["cost"]
    assert out["price_usd_per_million_tokens"] is None and out["per_1000_searches_usd"] is None  # "good" is not a priced model


def test_a_provider_without_vectors_or_a_key_is_skipped_with_the_reason(world):
    empty = FakeHosted("empty", {})
    nokey = FakeHosted("nokey", {}, key="")
    out = evaluate(world, [empty, nokey, world.good])["providers"]
    assert "no vectors for" in out[empty.key]["skipped"] and "reprocess --embed-provider gemini" in out[empty.key]["skipped"]
    assert "not configured" in out[nokey.key]["skipped"]
    assert "modes" in out[world.good.key]  # the others still ran
    assert empty.queries == [] and nokey.queries == []  # a skipped provider is never called


def test_a_provider_that_fails_during_the_run_is_invalid_not_scored_as_keyword_only(world):
    flaky = FakeHosted("flaky", {q: v for q, (_, v) in QUERIES.items()}, fail=EmbeddingUnavailable("down"))
    vectorstore.store(world.db, flaky, [(c.id, "native", unit(1, 0, 0, 0)) for c in world.db.query(Chunk).all()])
    world.db.commit()
    out = evaluate(world, [flaky])["providers"][flaky.key]
    assert "invalid run" in out["skipped"] and "keyword-only" in out["skipped"] and "modes" not in out


def test_partial_coverage_is_reported(world):
    partial = FakeHosted("partial", {q: v for q, (_, v) in QUERIES.items()})
    first = world.db.query(Chunk).order_by(Chunk.idx).first()
    vectorstore.store(world.db, partial, [(first.id, "native", unit(1, 0, 0, 0))])
    world.db.commit()
    messages = []
    c = costs.Corpus(videos=1, hours=1.0, native_chars=4000, en_chars=0, chunks=3)
    out = rpe.run_provider_eval(world.db, world.queries, [partial], k=5, repeats=1, cache=world.cache, corpus=c, video_ids=world.ids, progress=messages.append)
    assert out["providers"][partial.key]["coverage"] == {"have": 1, "want": 6}  # 3 chunks x (native + English)
    assert any("WARNING only 1/6" in m for m in messages)


def test_corpus_stats_read_hours_and_text_from_the_database(world):
    c = rpe.corpus_stats(world.db, world.ids)
    assert (c.videos, c.chunks, c.hours) == (1, 3, 1.0)
    assert c.native_chars == sum(len(f"text number {i}") for i in range(3)) and c.en_chars == sum(len(f"english {i}") for i in range(3))


def test_the_report_has_the_tables_the_numbers_and_the_caveats(world):
    results = evaluate(world, [world.good, FakeHosted("nokey", {}, key="")])
    text = rpe.render_report(results, k=5, n_queries=3, corpus=costs.Corpus(1, 1.0, 4000, 1000, 3),
                             transcriber_rows=rpe.transcription_rows(88.0, text_chars_per_hour=50_000), assume_both_variants=False,
                             text_chars_per_hour=50_000)
    assert "### Accuracy and warm latency (3 queries, k=5)" in text and "Recall@5" in text and "MRR@5" in text
    assert world.good.key in text and "100.0%" in text and "Median cold search" in text and "USD / 1,000 searches" in text
    # one row per TEXT_VARIANT_PROVIDER; 50,000 characters per hour
    assert "(TEXT_VARIANT_PROVIDER=sarvam) | INR 230 (about $2.61 at INR 88/USD)" in text  # 30 + 2 x 5 x 20
    assert "(TEXT_VARIANT_PROVIDER=none) | INR 30 (about $0.34 at INR 88/USD)" in text
    assert "sarvam + gemini-3.1-flash-lite for roman and English (TEXT_VARIANT_PROVIDER=gemini) | INR 30 + $0.059, about INR 35 in all at INR 88/USD" in text
    assert "ASSUMED token counts" in text and "https://ai.google.dev/gemini-api/docs/pricing" in text
    assert "ONE transcription job" in text and "50,000 characters per hour" in text and "no per-hour charge" in text
    assert "nokey" in text and "skipped, not configured" in text
    assert "estimates" in text and "Indic scripts" in text


# ---------------------------------------------------------------- the command line
def test_main_asks_before_making_api_calls_and_no_means_none_are_made(world, capsys):
    with patch.object(embedders, "make_embedder", side_effect=lambda n, **k: world.good if n == "gemini" else world.poor):
        code = rpe.main(["--queries", str(world.gold), "--providers", "gemini", "--out", str(world.gold.parent)], confirm=lambda p: "n")
    assert code == 1 and "Cancelled: no API calls were made" in capsys.readouterr().out
    assert world.good.queries == []


def test_main_runs_saves_json_and_reports_the_number_of_calls_it_will_make(world, capsys, tmp_path):
    with patch.object(embedders, "make_embedder", return_value=world.good):
        code = rpe.main(["--queries", str(world.gold), "--providers", "gemini", "--yes", "--k", "5", "--repeats", "1", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "Real API calls: about 3 query embeddings (3 x 1 hosted provider(s))" in out
    (saved,) = list(tmp_path.glob("providers-*.json"))
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data["config"]["k"] == 5 and world.good.key in data["providers"] and data["providers"][world.good.key]["modes"]["vector"]["summary"]["mrr"] == 1.0


def test_main_rejects_unknown_providers_and_a_missing_gold_file(world, capsys):
    assert rpe.main(["--queries", str(world.gold), "--providers", "llama"]) == 1
    assert "Unknown provider" in capsys.readouterr().out
    assert rpe.main(["--queries", str(world.gold.parent / "nope.jsonl")]) == 1
    assert "does not exist" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        rpe.main(["--queries", str(world.gold), "--k", "3"])  # Recall@5 needs k >= 5


def test_transcription_rows_without_a_rate_keep_gemini_in_usd_and_never_invent_one():
    rows = dict(rpe.transcription_rows(None, text_chars_per_hour=50_000))
    gem = next(v for k, v in rows.items() if "TEXT_VARIANT_PROVIDER=gemini" in k)
    assert gem == "INR 30 + $0.059" and "at INR" not in gem
    unknown = dict(rpe.transcription_rows(None, text_model="gemini-9-imaginary"))
    assert "unknown" in next(v for k, v in unknown.items() if "TEXT_VARIANT_PROVIDER=gemini" in k)


def test_the_text_model_price_table_has_sources_and_the_pinned_default_is_priced():
    from app.core.config import Settings
    assert pricing.text_model_price(Settings.model_fields["TEXT_VARIANT_MODEL"].default) is not None  # the default can be costed
    for model, p in pricing.TEXT_MODEL_PRICES.items():
        assert p.source.startswith("https://ai.google.dev/") and 0 < p.usd_in_per_million < p.usd_out_per_million, model
    assert pricing.text_model_cost_usd("gemini-3.1-flash-lite", 1_000_000, 1_000_000) == pytest.approx(0.25 + 1.50)
    assert pricing.text_model_cost_usd("gemini-3.1-flash-lite", 2_000_000, 0) == pytest.approx(0.50)
    assert pricing.text_model_cost_usd("not-a-model", 1, 1) is None


def test_the_estimated_cost_per_hour_of_video_for_every_provider():
    g = costs.text_variants_per_hour("gemini", "gemini-3.1-flash-lite", None, 88.0)
    assert (g["tokens_in"], g["tokens_out"]) == (36_100, 55_770) and g["usd"] == pytest.approx(0.0927, abs=1e-4)
    assert g["inr"] == pytest.approx(g["usd"] * 88)
    assert costs.text_variants_per_hour("gemini", "gemini-3.1-flash-lite")["inr"] is None  # no rate given: no INR figure
    assert costs.text_variants_per_hour("sarvam")["inr"] == pytest.approx(312.0)
    assert costs.text_variants_per_hour("none") == {"provider": "none", "model": None, "inr": 0.0, "usd": 0.0, "known": True, "tokens_in": 0, "tokens_out": 0}
    unknown = costs.text_variants_per_hour("gemini", "gemini-9-imaginary", None, 88.0)
    assert unknown["known"] is False and unknown["usd"] is None and unknown["inr"] is None
    # a pricier model costs more per hour for the same text
    assert costs.text_variants_per_hour("gemini", "gemini-3.5-flash-lite")["usd"] > g["usd"] > costs.text_variants_per_hour("gemini", "gemini-2.5-flash-lite")["usd"]
