"""Re-process existing videos with a different transcriber and/or embedding provider, without re-uploading them.

  python -m app.scripts.reprocess --dry-run --transcriber sarvam --embed-provider gemini     # the plan and its estimated cost
  python -m app.scripts.reprocess --transcriber sarvam --embed-provider gemini               # do it (asks before spending money)
  python -m app.scripts.reprocess --embed-provider cohere --video-ids <uuid>,<uuid>          # only re-embed (no transcription) those videos
  python -m app.scripts.reprocess --embed-provider local                                     # fill the local vectors that are missing

What happens to each video:
  transcribe   its current engine (videos.transcriber, NULL = whisper) is not the requested one: a normal ingest job is queued
               (download, transcribe, chunk, embed), so the admin page shows progress like for an upload. Old chunks are replaced
               only when the new ones are complete.
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
from app.services import embedders, jobevents, pricing, vectorstore
from app.services.embedders import estimate_tokens

BATCH = 64  # texts per embedding call (and per commit) when only embedding


@dataclass
class Plan:
    video_id: uuid.UUID
    title: str
    action: str  # transcribe | embed | up-to-date | busy
    detail: str = ""
    audio_seconds: float = 0.0  # transcribe: audio to be transcribed
    texts: int = 0  # embed: texts to embed
    tokens: int = 0  # embed: estimated tokens


def current_engine(video) -> str:
    return video.transcriber or "whisper"


def _missing_local(db, video_id) -> list[tuple[uuid.UUID, str, str]]:
    rows = db.query(Chunk).filter(Chunk.video_id == video_id, Chunk.embedding.is_(None)).order_by(Chunk.idx).all()
    return [(c.id, "native", (c.text_native or c.text).strip()) for c in rows if (c.text_native or c.text).strip()]


def missing_for(db, embedder, video_id) -> list[tuple[uuid.UUID, str, str]]:
    if embedder.uses_legacy_column:
        return _missing_local(db, video_id)
    return vectorstore.missing(db, embedder, [video_id])


def make_plan(db, video, *, transcriber: str | None, embedder, busy: set) -> Plan:
    if video.id in busy:
        return Plan(video.id, video.title, "busy", "a processing job is running for it")
    if transcriber and current_engine(video) != transcriber:
        return Plan(video.id, video.title, "transcribe", f"{current_engine(video)} -> {transcriber}", audio_seconds=float(video.duration_sec or 0))
    gaps = missing_for(db, embedder, video.id)
    if gaps:
        return Plan(video.id, video.title, "embed", f"{len(gaps)} text(s) without a {embedder.key} vector", texts=len(gaps),
                    tokens=sum(estimate_tokens(t) for _, _, t in gaps))
    return Plan(video.id, video.title, "up-to-date")


def estimate(plans: list[Plan], transcriber: str | None, embedder) -> dict:
    """Rough cost of the plan: Sarvam INR (one speech-to-text job per video plus the text APIs for the romanised and English versions),
    embedding USD for the texts to embed."""
    audio = sum(p.audio_seconds for p in plans if p.action == "transcribe")
    inr = pricing.sarvam_cost_inr(audio, variants=len(settings.sarvam_text_variants)) if transcriber == "sarvam" else 0.0
    tokens = sum(p.tokens for p in plans if p.action == "embed")
    price = pricing.embedding_price(embedder.key)
    usd = pricing.embedding_cost_usd(tokens, price) if price else None
    return {"audio_hours": audio / 3600, "sarvam_inr": inr, "embed_tokens": tokens, "embed_usd": usd, "embed_price_known": price is not None}


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
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the estimated cost, change nothing")
    ap.add_argument("--yes", action="store_true", help="do not ask before spending money")
    args = ap.parse_args(argv)
    try:
        ids = parse_ids(args.video_ids)
    except ValueError as exc:
        ap.error(f"bad video id: {exc}")

    embedder = embedder or embedders.get_embedder(args.embed_provider)
    if not args.dry_run:
        try:
            embedder.check_configured()  # a missing key should stop the run now, not after the first video
        except embedders.EmbeddingNotConfigured as exc:
            print(f"{exc}: nothing was changed.")
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
        plans = [make_plan(db, v, transcriber=args.transcriber, embedder=embedder, busy=busy) for v in videos]
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
        if est["embed_tokens"]:
            cost = f"about ${est['embed_usd']:.4f}" if est["embed_price_known"] else "price unknown for this model"
            print(f"to embed now: ~{est['embed_tokens']:,} tokens -> {cost}")
        if args.dry_run:
            print("Dry run: nothing was changed.")
            return 0
        work = [p for p in plans if p.action in ("transcribe", "embed")]
        if not work:
            print("Nothing to do.")
            return 0
        spends = (any(p.action == "transcribe" for p in work) and args.transcriber == "sarvam") or \
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
