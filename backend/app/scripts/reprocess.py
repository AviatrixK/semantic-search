"""Re-process existing videos with a different transcriber and/or embedding provider, without re-uploading them.

  python -m app.scripts.reprocess --dry-run --transcriber sarvam --embed-provider gemini     # the plan and its estimated cost
  python -m app.scripts.reprocess --transcriber sarvam --embed-provider gemini               # do it (asks before spending money)
  python -m app.scripts.reprocess --embed-provider cohere --video-ids <uuid>,<uuid>          # only re-embed (no transcription) those videos
  python -m app.scripts.reprocess --embed-provider local                                     # fill the local vectors that are missing

What happens to each video:
  transcribe   its current engine (videos.transcriber, NULL = whisper) is not the requested one: a normal ingest job is queued
               (download, transcribe, chunk, embed), so the admin page shows progress like for an upload. Old chunks are replaced
               only when the new ones are complete.
  text         the video was transcribed but the English / romanised text of some chunks is still missing (videos.text_variants_status is
               "pending": the translation service was down or answered badly during ingest). ONLY the chunks that lack a variant are
               sent to TEXT_VARIANT_MODEL, then the vectors that depend on them are embedded. Nothing is sent twice.
               --fill-translations extends this to EVERY video whose non-English chunks lack a variant, whatever its status (videos
               ingested before translation existed, or transcribed by Whisper): the video page transcript needs text_en / text_roman.
               No audio is touched, nothing is re-transcribed; chunks already in Latin letters are never sent.
  embed        the transcript is fine, but this embedding provider has no vectors for some of its text yet: only those are embedded
               here, in batches, each batch committed (the cost is tokens, no audio).
  up-to-date   nothing to do.   busy: a job is running for it, left alone.

RESUMABLE: every decision comes from the database (the engine recorded on the video, the vectors already stored), so after a crash,
a Ctrl-C or a failed batch you just run the same command again and only the remaining work is done. Failed batches never leave
partial vectors for a video that matters: a re-run fills the gaps. Exit code 1 if any video failed.

--embed-provider also decides which provider the queued transcription jobs embed with (default: EMBED_PROVIDER). Estimates use list
prices (app/services/pricing.py) and a rough token count: check the provider's dashboard for what was really billed."""
import argparse
import sys
import uuid
from dataclasses import dataclass

from sqlalchemy import select

from app.core.config import settings
from app.core.db import SessionLocal, engine
from app.models import Chunk, Job, Video
from app.services import embedders, jobevents, pricing, textvariants, vectorstore
from app.services.embedders import estimate_tokens

BATCH = 64  # texts per embedding call (and per commit) when only embedding


@dataclass
class Plan:
    video_id: uuid.UUID
    title: str
    action: str  # transcribe | text | embed | up-to-date | busy
    detail: str = ""
    audio_seconds: float = 0.0  # transcribe: audio to be transcribed
    texts: int = 0  # embed: texts to embed
    tokens: int = 0  # embed: estimated tokens
    text_chunks: int = 0  # text: chunks still without their English / romanised text
    text_chars: int = 0  # text: characters of those chunks


def current_engine(video) -> str:
    return video.transcriber or "whisper"


def _missing_local(db, video_id) -> list[tuple[uuid.UUID, str, str]]:
    rows = db.query(Chunk).filter(Chunk.video_id == video_id, Chunk.embedding.is_(None)).order_by(Chunk.idx).all()
    return [(c.id, "native", (c.text_native or c.text).strip()) for c in rows if (c.text_native or c.text).strip()]


def missing_for(db, embedder, video_id) -> list[tuple[uuid.UUID, str, str]]:
    if embedder.uses_legacy_column:
        return _missing_local(db, video_id)
    return vectorstore.missing(db, embedder, [video_id])


def make_plan(db, video, *, transcriber: str | None, embedder, busy: set, fill_translations: bool = False) -> Plan:
    if video.id in busy:
        return Plan(video.id, video.title, "busy", "a processing job is running for it")
    if transcriber and current_engine(video) != transcriber:
        return Plan(video.id, video.title, "transcribe", f"{current_engine(video)} -> {transcriber}", audio_seconds=float(video.duration_sec or 0))
    if (fill_translations or getattr(video, "text_variants_status", None) == "pending") and settings.TEXT_VARIANT_PROVIDER == "gemini":
        todo = textvariants.chunks_needing_variants(db, video.id)
        if fill_translations:  # only chunks that cost a model call: a Latin-script chunk needs no translation for the transcript
            todo = [c for c in todo if textvariants.needs_model((c.text_native or c.text).strip())]
        if todo:
            return Plan(video.id, video.title, "text", f"{len(todo)} chunk(s) without English/romanised text", text_chunks=len(todo),
                        text_chars=sum(len((c.text_native or c.text).strip()) for c in todo))
    gaps = missing_for(db, embedder, video.id)
    if gaps:
        return Plan(video.id, video.title, "embed", f"{len(gaps)} text(s) without a {embedder.key} vector", texts=len(gaps),
                    tokens=sum(estimate_tokens(t) for _, _, t in gaps))
    return Plan(video.id, video.title, "up-to-date")


def estimate(plans: list[Plan], transcriber: str | None, embedder) -> dict:
    """Rough cost of the plan. Sarvam INR: ONE speech-to-text job per video, plus its text APIs when TEXT_VARIANT_PROVIDER=sarvam. Text
    variants from Gemini are priced in USD from an ESTIMATED token count (pricing.py); embeddings in USD from the texts to embed."""
    audio = sum(p.audio_seconds for p in plans if p.action == "transcribe")
    provider = settings.TEXT_VARIANT_PROVIDER
    inr = 0.0
    if transcriber == "sarvam":
        inr = pricing.sarvam_cost_inr(audio, variants=0)
        if provider == "sarvam":
            inr = pricing.sarvam_cost_inr(audio, variants=len(settings.sarvam_text_variants))
    tokens = sum(p.tokens for p in plans if p.action == "embed")
    price = pricing.embedding_price(embedder.key)
    usd = pricing.embedding_cost_usd(tokens, price) if price else None
    # text variants made by Gemini: for the videos about to be transcribed (by audio length) and for the pending chunks
    text_in = text_out = 0
    if provider == "gemini":
        if transcriber == "sarvam" and audio:
            per_hour = pricing.text_variants_per_hour("gemini", settings.TEXT_VARIANT_MODEL)
            text_in += round(per_hour["tokens_in"] * audio / 3600)
            text_out += round(per_hour["tokens_out"] * audio / 3600)
        for p in plans:
            if p.action == "text":
                i, o = pricing.estimate_text_variant_tokens(p.text_chars, p.text_chunks)
                text_in, text_out = text_in + i, text_out + o
    text_usd = pricing.text_model_cost_usd(settings.TEXT_VARIANT_MODEL, text_in, text_out) if (text_in or text_out) else 0.0
    return {"audio_hours": audio / 3600, "sarvam_inr": inr, "embed_tokens": tokens, "embed_usd": usd, "embed_price_known": price is not None,
            "text_tokens_in": text_in, "text_tokens_out": text_out, "text_usd": text_usd, "text_price_known": text_usd is not None}


def text_cost_lines(inr_per_usd: float | None) -> list[str]:
    """The cost of making text_en / text_roman per hour of video for each TEXT_VARIANT_PROVIDER (the configured one is marked)."""
    out = []
    for provider in ("gemini", "sarvam", "none"):
        model = settings.TEXT_VARIANT_MODEL if provider == "gemini" else None
        r = pricing.text_variants_per_hour(provider, model, None, inr_per_usd)
        if provider == "gemini":
            what = (f"${r['usd']:.3f}" + (f" (about INR {r['inr']:.1f})" if r["inr"] is not None else "") + f", ~{r['tokens_in']:,} input + ~{r['tokens_out']:,} output tokens, "
                    f"ESTIMATED, list price {pricing.GEMINI_PRICING_SOURCE}") if r["known"] else f"unknown: no price for {model} in pricing.py"
        elif provider == "sarvam":
            what = f"INR {r['inr']:.0f} ({pricing.SARVAM_TEXT_INR_PER_10K_CHARS:.0f} per 10K characters x 2 versions, {pricing.SARVAM_SOURCE})"
        else:
            what = "free (native text only)"
        out.append(f"  TEXT_VARIANT_PROVIDER={provider:<7}{'(current) ' if provider == settings.TEXT_VARIANT_PROVIDER else '          '}"
                   f"per hour of video: {what}")
    return out


def embed_missing(db, embedder, video_id, *, batch: int | None = None) -> int:
    """Embeds the texts of one video that have no vector for this embedder yet, `batch` at a time, committing each batch (so an
    interruption loses at most one batch and the next run continues). Returns how many texts were embedded."""
    batch = batch or BATCH
    vectorstore.ensure_index(engine, embedder)
    done = 0
    gaps = missing_for(db, embedder, video_id)
    for i in range(0, len(gaps), batch):
        part = gaps[i:i + batch]
        vectors = embedder.embed_documents([t for _, _, t in part])
        if embedder.uses_legacy_column:
            for (cid, _f, _t), v in zip(part, vectors):
                db.get(Chunk, cid).embedding = v
        else:
            vectorstore.store(db, embedder, [(cid, f, v) for (cid, f, _t), v in zip(part, vectors)])
        db.commit()
        done += len(part)
    return done


def queue_transcription(db, video, transcriber: str, embed_provider: str | None, enqueue) -> uuid.UUID:
    job = Job(video_id=video.id)
    video.status = "uploaded"  # shown as "Processing" until the new chunks replace the old ones
    db.add(job)
    db.commit()
    jobevents.publish(jobevents.payload(job.id, video.id, "queued", 0))
    enqueue(str(video.id), str(job.id), transcriber, embed_provider)
    return job.id


def _default_enqueue(video_id, job_id, transcriber, embed_provider):
    from app.workers.tasks import ingest_video  # local import: keeps the script light for --dry-run
    ingest_video.delay(video_id, job_id, transcriber, embed_provider)


def parse_ids(values: list[str] | None) -> list[uuid.UUID] | None:
    if not values:
        return None
    out = []
    for token in (t for v in values for t in v.replace(",", " ").split()):
        out.append(uuid.UUID(token))  # ValueError for a malformed id: the caller reports it
    return out


def main(argv: list[str] | None = None, *, enqueue=None, embedder=None, confirm=input) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video-ids", nargs="+", metavar="ID", help="only these videos (space or comma separated)")
    ap.add_argument("--transcriber", choices=["whisper", "sarvam"], help="re-transcribe videos not yet done by this engine")
    ap.add_argument("--embed-provider", choices=list(embedders.PROVIDERS), help="embedding provider to fill (default: EMBED_PROVIDER)")
    ap.add_argument("--inr-per-usd", type=float, help="exchange rate, only used to show the Gemini cost in INR too (none is assumed)")
    ap.add_argument("--fill-translations", action="store_true",
                    help="also make the missing English / romanised text of videos that are not marked pending (for the video page transcript)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the estimated cost, change nothing")
    ap.add_argument("--yes", action="store_true", help="do not ask before spending money")
    args = ap.parse_args(argv)
    try:
        ids = parse_ids(args.video_ids)
    except ValueError as exc:
        ap.error(f"bad video id: {exc}")

    if args.fill_translations and settings.TEXT_VARIANT_PROVIDER != "gemini":
        print(f"--fill-translations needs TEXT_VARIANT_PROVIDER=gemini (it is {settings.TEXT_VARIANT_PROVIDER!r}): nothing was changed.")
        return 1
    embedder = embedder or embedders.get_embedder(args.embed_provider)
    if not args.dry_run:
        try:
            embedder.check_configured()  # a missing key should stop the run now, not after the first video
        except embedders.EmbeddingNotConfigured as exc:
            print(f"{exc}: nothing was changed.")
            return 1
        if args.fill_translations and not settings.GEMINI_API_KEY:
            print("GEMINI_API_KEY is empty: nothing was changed.")
            return 1
        if args.transcriber == "sarvam" and not settings.SARVAM_API_KEY:
            print("SARVAM_API_KEY is empty: nothing was changed.")
            return 1

    with SessionLocal() as db:
        stmt = select(Video).order_by(Video.created_at)
        if ids:
            stmt = stmt.where(Video.id.in_(ids))
        videos = db.scalars(stmt).all()
        if ids and len(videos) != len(set(ids)):
            print("Unknown video id(s): " + ", ".join(str(i) for i in set(ids) - {v.id for v in videos}))
            return 1
        busy = set(db.scalars(select(Job.video_id).where(Job.stage.in_(jobevents.ACTIVE_STAGES))).all())
        plans = [make_plan(db, v, transcriber=args.transcriber, embedder=embedder, busy=busy,
                           fill_translations=args.fill_translations) for v in videos]
        by_video = {v.id: v for v in videos}

        for p in plans:
            print(f"{p.video_id}  {p.title[:36]!s:36}  {p.action:11} {p.detail}")
        counts: dict[str, int] = {}
        for p in plans:
            counts[p.action] = counts.get(p.action, 0) + 1
        est = estimate(plans, args.transcriber, embedder)
        print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f"   (embedder {embedder.key})")
        if est["audio_hours"]:
            line = f"to transcribe: {est['audio_hours']:.2f} h of audio"
            if args.transcriber == "sarvam":
                line += (f" -> about INR {est['sarvam_inr']:.0f}: one transcription job at INR {pricing.SARVAM_STT_INR_PER_HOUR:.0f}/h plus the text APIs for "
                         f"{'+'.join(settings.sarvam_text_variants) or 'no extra'} at INR {pricing.SARVAM_TEXT_INR_PER_10K_CHARS:.0f} per 10K characters, "
                         f"assuming ~{pricing.TEXT_CHARS_PER_AUDIO_HOUR:,} characters per hour ({pricing.SARVAM_SOURCE})")
            print(line + "; their embeddings come on top when the jobs run")
        if est["text_tokens_in"] or est["text_tokens_out"]:
            cost = f"about ${est['text_usd']:.4f}" if est["text_price_known"] else f"price unknown for {settings.TEXT_VARIANT_MODEL}"
            print(f"to translate/romanise now ({settings.TEXT_VARIANT_MODEL}): ~{est['text_tokens_in']:,} input + ~{est['text_tokens_out']:,} output tokens (estimated) -> {cost}")
        if args.dry_run or args.transcriber == "sarvam":
            print("text_en / text_roman cost per hour of video:")
            print("\n".join(text_cost_lines(args.inr_per_usd)))
        if est["embed_tokens"]:
            cost = f"about ${est['embed_usd']:.4f}" if est["embed_price_known"] else "price unknown for this model"
            print(f"to embed now: ~{est['embed_tokens']:,} tokens -> {cost}")
        if args.dry_run:
            print("Dry run: nothing was changed.")
            return 0
        work = [p for p in plans if p.action in ("transcribe", "text", "embed")]
        if not work:
            print("Nothing to do.")
            return 0
        spends = (any(p.action == "transcribe" for p in work) and args.transcriber == "sarvam") or any(p.action == "text" for p in work) or \
                 (any(p.action == "embed" for p in work) and embedder.provider != "local")
        if spends and not args.yes:
            try:
                answer = confirm("This uses paid APIs. Continue? [y/N] ")
            except EOFError:
                answer = ""
            if answer.strip().lower() not in ("y", "yes"):
                print("Cancelled: nothing was changed. (Use --yes to skip this question.)")
                return 1

        failed = 0
        enqueue = enqueue or _default_enqueue
        for p in work:
            video = by_video[p.video_id]
            try:
                if p.action == "transcribe":
                    job_id = queue_transcription(db, video, args.transcriber, args.embed_provider, enqueue)
                    print(f"{p.video_id}  queued job {job_id}")
                elif p.action == "text":
                    res = textvariants.fill_video(db, video)
                    n = embed_missing(db, embedder, p.video_id)  # the English vectors that were waiting for the English text
                    left = f", {len(res.pending)} still pending (run again)" if res.pending else ""
                    print(f"{p.video_id}  text variants made for {len(res.results)} chunk(s) ({res.usage.input:,} in / {res.usage.output:,} out tokens), "
                          f"embedded {n} text(s){left}")
                    failed += 1 if res.aborted else 0
                else:
                    n = embed_missing(db, embedder, p.video_id)
                    print(f"{p.video_id}  embedded {n} text(s)")
            except Exception as exc:  # one failing video must not stop the others; a re-run picks it up again
                db.rollback()
                failed += 1
                print(f"{p.video_id}  FAILED: {type(exc).__name__}: {str(exc)[:160]}")
        tokens, guessed = getattr(embedder, "tokens_used", 0), getattr(embedder, "tokens_estimated", 0)
        print(f"\nDone{', ' + str(failed) + ' failed (run the same command again to retry them)' if failed else ''}. "
              f"Embedding tokens used now: {tokens:,}" + (f" ({guessed:,} of them estimated: the provider reported no usage)." if guessed else " (as reported by the provider)." if tokens else "."))
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
