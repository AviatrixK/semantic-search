"""Compare embedding providers on the gold queries: Recall@1/@5, MRR, median search latency and estimated cost.

    docker compose exec api python eval/run_provider_eval.py                       # every provider that is configured and has vectors
    docker compose exec api python eval/run_provider_eval.py --providers local,gemini,cohere --repeats 3
    docker compose exec api python eval/run_provider_eval.py --inr-per-usd 88 --assume-both-variants

Each provider is evaluated on ITS OWN stored vectors, so first fill them (they live side by side in the database):
    docker compose exec api python -m app.scripts.reprocess --embed-provider gemini --dry-run     # then without --dry-run

This makes REAL API calls to the hosted providers: one query embedding per gold query and provider (repeats are served from an
in-memory cache, so they are free and measure the warm path). It shows the number of calls and asks first (--yes skips the question).
The gold file is the developer's: nothing here invents queries.

Latency: `cold` = the first, uncached search (includes the provider's network round trip); `warm` = the median of repeated cached
searches. A provider that fails or times out DURING the run is reported as invalid, not scored: search would silently fall back to
keyword-only for it, which would make its numbers meaningless.

Cost: estimated from list prices (app/services/pricing.py) and a rough token count; see eval/costs.py for what is and is not included.
Saves eval/results/providers-<timestamp>.json."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `app` and `eval` import when run as a script

from eval import costs, metrics  # noqa: E402
from eval.run_eval import ModeUnavailable, run_retrieval_eval  # noqa: E402

MODES = ("vector", "hybrid")


class MemoryCache:
    """A Redis stand-in for the run: queries are 'cold' once, then cached, and the cache can be emptied between providers."""

    def __init__(self):
        self.data: dict = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value

    def clear(self):
        self.data.clear()


def coverage(db, embedder, video_ids: list) -> tuple[int, int]:
    """(texts that have a vector for this embedder, texts that should have one) over the given videos."""
    from sqlalchemy import select

    from app.models import Chunk
    from app.services import vectorstore
    chunks = db.scalars(select(Chunk).where(Chunk.video_id.in_(video_ids))).all()
    if embedder.uses_legacy_column:
        want = sum(1 for c in chunks if (c.text_native or c.text).strip())
        return sum(1 for c in chunks if c.embedding is not None), want
    want = sum(len(vectorstore.field_texts(c)) for c in chunks)
    return want - len(vectorstore.missing(db, embedder, video_ids)), want


def not_ready_reason(db, embedder, video_ids: list) -> str | None:
    """Why this provider cannot be evaluated, or None."""
    from app.services.embedders import EmbeddingNotConfigured
    try:
        embedder.check_configured()
    except EmbeddingNotConfigured as exc:
        return f"not configured ({exc})"
    have, want = coverage(db, embedder, video_ids)
    if have == 0:
        return f"no vectors for {embedder.key} yet: run `python -m app.scripts.reprocess --embed-provider {embedder.provider}`"
    return None


def provider_modes(embedder, which=MODES) -> dict:
    """mode name -> fn(db, query, k) -> hits, searching with THIS embedder. Raises ModeUnavailable if the provider failed during the
    call (search would then have silently answered with keywords only)."""
    from app.services import retrieval
    from app.services.timing import Timings

    def make(mode):
        def search(db, q, k):
            t = Timings()
            run = retrieval.vector_search if mode == "vector" else retrieval.hybrid_search
            kwargs = {} if mode == "vector" else {"rerank": False}
            hits = run(db, q, k=k, highlight=False, timings=t, embedder=embedder, **kwargs)
            if t.counts.get("embed_failed"):
                raise ModeUnavailable(f"{embedder.provider} failed or timed out during the run: these numbers would be keyword-only")
            return hits
        return search

    return {m: make(m) for m in which}


def corpus_stats(db, video_ids: list) -> costs.Corpus:
    from sqlalchemy import func, select

    from app.models import Chunk, Video
    row = db.execute(select(func.count(Video.id), func.coalesce(func.sum(Video.duration_sec), 0)).where(Video.id.in_(video_ids))).one()
    c = db.execute(select(func.count(Chunk.id), func.coalesce(func.sum(func.length(func.coalesce(Chunk.text_native, Chunk.text))), 0),
                          func.coalesce(func.sum(func.length(func.coalesce(Chunk.text_en, ""))), 0)).where(Chunk.video_id.in_(video_ids))).one()
    return costs.Corpus(videos=row[0], hours=float(row[1]) / 3600, native_chars=int(c[1]), en_chars=int(c[2]), chunks=int(c[0]))


def run_provider_eval(db, queries: list, embedder_list: list, *, k: int, repeats: int, cache: MemoryCache, corpus: costs.Corpus,
                      video_ids: list, clock=time.perf_counter, progress=print, assume_both_variants: bool = False,
                      modes: tuple = MODES) -> dict:
    out: dict = {"providers": {}}
    avg_estimate = costs.average_query_tokens([q.query for q in queries])
    for emb in embedder_list:
        entry: dict = {"provider": emb.provider, "model": emb.model, "dim": emb.dim, "key": emb.key}
        out["providers"][emb.key] = entry
        reason = not_ready_reason(db, emb, video_ids)
        if reason:
            entry["skipped"] = reason
            progress(f"{emb.key}: skipped ({reason})")
            continue
        have, want = coverage(db, emb, video_ids)
        entry["coverage"] = {"have": have, "want": want}
        if have < want:
            progress(f"{emb.key}: WARNING only {have}/{want} texts have vectors; recall is understated until reprocess finishes")
        cache.clear()
        tokens_before, estimated_before = emb.tokens_used, getattr(emb, "tokens_estimated", 0)
        searches = provider_modes(emb, modes)
        try:
            cold = []
            for q in queries:  # the first search of each query is uncached: it pays the provider round trip
                started = clock()
                searches[modes[0]](db, q.query, k)
                cold.append((clock() - started) * 1000)
                if hasattr(db, "rollback"):
                    db.rollback()
            entry["cold_median_ms"] = metrics.median(cold)
            reported = (emb.tokens_used - tokens_before) / len(queries) if queries else 0
            guessed = (getattr(emb, "tokens_estimated", 0) - estimated_before) / len(queries) if queries else 0
            # the provider's own count wherever its responses carried one; the estimate only when none did
            entry["query_tokens"] = {"avg": reported if reported > 0 else avg_estimate,
                                     "source": ("estimated (4 chars/token)" if reported <= 0 else "reported by the provider" if guessed <= 0
                                                else "partly estimated (some responses carried no usage)")}
            result = run_retrieval_eval(db, queries, searches, k=k, repeats=repeats, clock=clock, progress=lambda m: None)
        except ModeUnavailable as exc:
            entry["skipped"] = f"invalid run: {exc}"
            progress(f"{emb.key}: {entry['skipped']}")
            continue
        entry["modes"], entry["skipped_modes"] = result["modes"], result["skipped"]
        if result["skipped"]:
            entry["skipped"] = "invalid run: " + "; ".join(result["skipped"].values())
            continue
        cost = costs.cost_row(emb.key, entry["query_tokens"]["avg"], corpus, assume_both_variants=assume_both_variants)
        entry["cost"] = {"per_1000_searches_usd": cost.per_1000_searches_usd, "per_video_hour_usd": cost.per_video_hour_usd,
                         "price_usd_per_million_tokens": cost.price.usd_per_million_tokens if cost.price else None,
                         "price_source": cost.price.source if cost.price else None, "price_confirmed": cost.price.confirmed if cost.price else None}
        progress(f"{emb.key}: done")
    return out


def render_report(results: dict, *, k: int, n_queries: int, corpus: costs.Corpus, transcriber_rows: list[tuple[str, str]], assume_both_variants: bool,
                  text_chars_per_hour: float | None = None) -> str:
    from app.services import pricing
    text_chars_per_hour = text_chars_per_hour or pricing.TEXT_CHARS_PER_AUDIO_HOUR
    rows, lat = [], []
    for key, e in results["providers"].items():
        if "modes" not in e:
            continue
        for mode, m in e["modes"].items():
            s = m["summary"]
            rows.append([key, mode, metrics.fmt(s["recall@1"], "pct"), metrics.fmt(s["recall@5"], "pct"), metrics.fmt(s["mrr"], "ratio"),
                         metrics.fmt(s["median_latency_ms"], "ms")])
        c = e.get("cost") or {}
        price = f"{c['price_usd_per_million_tokens']:.2f}" + ("" if c.get("price_confirmed", True) else " (not on current pricing page)") if c.get("price_usd_per_million_tokens") is not None else "unknown"
        if c.get("price_usd_per_million_tokens") == 0:
            price = "free (local)"
        lat.append([key, metrics.fmt(e.get("cold_median_ms"), "ms"), f"{e['query_tokens']['avg']:.1f}", price,
                    costs.fmt_usd(c.get("per_1000_searches_usd")), costs.fmt_usd(c.get("per_video_hour_usd"))])
    parts = [f"### Accuracy and warm latency ({n_queries} queries, k={k})",
             metrics.markdown_table(["Provider / model @ dims", "Mode", "Recall@1", "Recall@5", f"MRR@{k}", "Median latency (warm)"], rows) if rows else "_no provider produced results_",
             "", "### Cold latency and cost (embedding only)",
             metrics.markdown_table(["Provider / model @ dims", "Median cold search", "Query tokens", "USD / 1M tokens", "USD / 1,000 searches", "USD / video hour"], lat) if lat else "_n/a_",
             "", f"Corpus used for the per-hour numbers: {corpus.videos} video(s), {corpus.hours:.2f} h, {corpus.chunks} chunks, "
                 f"{corpus.native_chars:,} native + {corpus.en_chars:,} English characters"
                 + (" (English variant assumed as long as the native one)." if assume_both_variants else ".")]
    parts += ["", "### Transcription cost per hour of video", metrics.markdown_table(["Engine", "Cost per hour of video"], [list(r) for r in transcriber_rows]),
              f"_Sarvam: ONE transcription job at INR {pricing.SARVAM_STT_INR_PER_HOUR:.0f} per hour of audio, billed per second, plus INR "
              f"{pricing.SARVAM_TEXT_INR_PER_10K_CHARS:.0f} per 10K characters for each text version made by Sarvam's text APIs (romanised, English); "
              f"about {text_chars_per_hour:,.0f} characters per hour of video ({pricing.SARVAM_SOURCE}). The Gemini row uses list prices "
              f"({pricing.GEMINI_PRICING_SOURCE}) and ASSUMED token counts (see pricing.py: not measured until videos have been processed; "
              f"videos.text_variants_in_tokens / _out_tokens hold the real ones)._"]
    for key, e in results["providers"].items():
        if e.get("skipped"):
            parts.append(f"\n_{key}: skipped, {e['skipped']}_")
    parts.append("\n_Token counts are estimates (about 4 characters per token) unless marked as reported; Indic scripts use more tokens per character, "
                 "so real costs for Hindi and similar are higher. Prices are list prices, see app/services/pricing.py for sources and dates._")
    return "\n".join(parts)


def transcription_rows(inr_per_usd: float | None, text_chars_per_hour: float | None = None, text_model: str | None = None) -> list[tuple[str, str]]:
    """Transcription cost per hour of video for every TEXT_VARIANT_PROVIDER. No exchange rate is assumed: without `inr_per_usd` the
    Gemini row stays in USD next to the INR for speech-to-text."""
    from app.core.config import settings
    from app.services import pricing
    model = text_model or settings.TEXT_VARIANT_MODEL
    stt = costs.transcription_per_hour_inr("sarvam", 0)
    usd_of = lambda inr: f" (about ${inr / inr_per_usd:.2f} at INR {inr_per_usd:g}/USD)" if inr_per_usd else ""
    rows = [("whisper (local)", "no per-hour charge (your CPU time)"),
            ("sarvam, native text only (TEXT_VARIANT_PROVIDER=none)", f"INR {stt:.0f}{usd_of(stt)}")]
    s = costs.text_variants_per_hour("sarvam", None, text_chars_per_hour, inr_per_usd)
    rows.append(("sarvam + its text APIs for roman and English (TEXT_VARIANT_PROVIDER=sarvam)", f"INR {stt + s['inr']:.0f}{usd_of(stt + s['inr'])}"))
    g = costs.text_variants_per_hour("gemini", model, text_chars_per_hour, inr_per_usd)
    if not g["known"]:
        value = f"INR {stt:.0f} + unknown (no price for {model} in app/services/pricing.py)"
    else:
        total = f", about INR {stt + g['inr']:.0f} in all at INR {inr_per_usd:g}/USD" if g["inr"] is not None else ""
        value = f"INR {stt:.0f} + ${g['usd']:.3f}{total}"
    rows.append((f"sarvam + {model} for roman and English (TEXT_VARIANT_PROVIDER=gemini)", value))
    return rows


def save_results(results: dict, config: dict, directory: Path = metrics.RESULTS_DIR, now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"providers-{now.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"timestamp": now.isoformat(), "config": config, **results}, indent=2, default=str), encoding="utf-8")
    return path


def main(argv: list[str] | None = None, *, confirm=input) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queries", type=Path, default=metrics.DEFAULT_QUERIES)
    ap.add_argument("--providers", help="comma separated: local,gemini,cohere,openai (default: all)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3, help="timed warm runs per query and mode (the median is used)")
    ap.add_argument("--yes", action="store_true", help="do not ask before making real API calls")
    ap.add_argument("--inr-per-usd", type=float, help="exchange rate to add Sarvam's INR price to the USD totals (none is assumed)")
    ap.add_argument("--assume-both-variants", action="store_true",
                    help="cost per video hour as if every chunk also had an English variant (what Sarvam adds to Whisper-era videos)")
    ap.add_argument("--out", type=Path, default=metrics.RESULTS_DIR)
    args = ap.parse_args(argv)
    if args.k < 5 or args.repeats < 1:
        ap.error("--k must be at least 5 and --repeats at least 1")

    if not args.queries.exists():
        print(f"{args.queries} does not exist. Create it with: docker compose exec api python eval/add_query.py")
        return 1
    try:
        queries = metrics.load_queries(args.queries)
    except ValueError as exc:
        print(f"Bad gold file: {exc}")
        return 1
    if not queries:
        print(f"{args.queries} has no queries yet.")
        return 1

    import uuid

    from app.core.config import settings
    from app.core.db import SessionLocal
    from app.services import embedders

    names = [p.strip() for p in (args.providers or ",".join(embedders.PROVIDERS)).split(",") if p.strip()]
    unknown = [n for n in names if n not in embedders.PROVIDERS]
    if unknown:
        print(f"Unknown provider(s): {', '.join(unknown)}. Use: {', '.join(embedders.PROVIDERS)}")
        return 1
    embedder_list = [embedders.make_embedder(n) for n in names]
    video_ids = [uuid.UUID(v) for v in sorted({g.video_id for q in queries for g in q.gold})]

    with SessionLocal() as db:
        runnable = [e for e in embedder_list if not_ready_reason(db, e, video_ids) is None]
        hosted = [e for e in runnable if not e.uses_legacy_column]
        calls = len(queries) * len(hosted)
        print(f"{len(queries)} gold queries; providers: {', '.join(e.key for e in embedder_list)}")
        print(f"Will run: {', '.join(e.key for e in runnable) or 'nothing'}. Real API calls: about {calls} query embeddings "
              f"({len(queries)} x {len(hosted)} hosted provider(s)); repeats are cached.")
        if hosted and not args.yes:
            try:
                answer = confirm("Continue? [y/N] ")
            except EOFError:
                answer = ""
            if answer.strip().lower() not in ("y", "yes"):
                print("Cancelled: no API calls were made. (--yes skips this question.)")
                return 1
        corpus = corpus_stats(db, video_ids)
        cache = MemoryCache()
        from app.services import embedders as _e
        _e._redis = lambda: cache  # queries are cached in memory for this run only: cold once, then warm
        results = run_provider_eval(db, queries, embedder_list, k=args.k, repeats=args.repeats, cache=cache, corpus=corpus, video_ids=video_ids,
                                    assume_both_variants=args.assume_both_variants)

    chars_per_hour = corpus.native_chars / corpus.hours if corpus.hours > 0 else None  # this library's text volume, else pricing's measured default
    print("\n" + render_report(results, k=args.k, n_queries=len(queries), corpus=corpus, transcriber_rows=transcription_rows(args.inr_per_usd, text_chars_per_hour=chars_per_hour),
                               assume_both_variants=args.assume_both_variants, text_chars_per_hour=chars_per_hour) + "\n")
    config = {"queries_file": str(args.queries), "k": args.k, "repeats": args.repeats, "min_score": {p: settings.min_score_for(p) for p in embedders.PROVIDERS},
              "embed_dimensions": settings.EMBED_DIMENSIONS, "sarvam_text_variants": settings.sarvam_text_variants, "inr_per_usd": args.inr_per_usd,
              "assume_both_variants": args.assume_both_variants, "videos": len(video_ids),
              "corpus": {"hours": corpus.hours, "chunks": corpus.chunks, "native_chars": corpus.native_chars, "en_chars": corpus.en_chars}}
    print("saved", save_results(results, config, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
