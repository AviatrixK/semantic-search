# Decisions

Why things are the way they are, including what was considered and rejected. Newest first. The short operational rules for
working in the repo live in `Claude.md`; this file keeps the reasoning.

## 2026-10 Round 1 local test (browser network recording)

### D6. The embedding model is loaded and warmed before the API reports ready

**Problem.** The first search took 11.7 s, all of it `embed;dur=11651`: `embedding._model()` was an `lru_cache` that loaded
`sentence-transformers` on the first request (and gunicorn runs two workers, each paying it separately).

**Decision.** In the FastAPI lifespan, after the bucket check, `embedding.warm_up()` loads the model and encodes one string (it runs
in a thread so the event loop stays alive). The lifespan runs once per gunicorn worker, which is what is needed because each worker
is its own process with its own copy of the model. Readiness has three layers:
- uvicorn serves no request before lifespan startup has finished, so a request can never reach a cold worker;
- `/health` returns 503 while `app.state.ready` is False (set False first thing in the lifespan, True after warm-up), so the
  compose healthcheck (`urlopen` raises on 503) and the deploy script wait for it;
- a failed warm-up (model missing, out of memory) raises and kills the worker, so the container is `unhealthy` and the cause is in
  the log, instead of an API that looks fine and fails on the first search.

`start_period` of the api healthcheck went from 90 s to 240 s and `deploy.sh` waits 5 minutes: with two workers importing torch and
loading the model at once on 2 vCPUs this is slow (45 s measured on a cold laptop). `WARMUP_ON_START=false` exists for development
shells and tests that do not want the model. Measured here: warm-up 45 s once, first query afterwards 14 ms.

**Thread limits stay out of the API.** `WHISPER_THREADS` and `TORCH_THREADS` are in the shared `.env.prod`, so the api container
sees them, but only worker code reads them (Whisper is never loaded in the api; `torch.set_num_threads` is called only from the
Celery `worker_process_init` hook). A test scans the source and fails if any other module calls `limit_torch_threads`, or if
another module starts reading those settings, so the API keeps all CPUs for query embedding.

**Rejected.**
- *`gunicorn --preload` (load in the master, fork the workers).* It would load once and share the weights copy-on-write, saving
  RAM and startup time, but: PyTorch's thread pools and OpenMP state are not fork-safe (children can hang or crash on the first
  encode); the lifespan, the S3 client and the asyncio loop must not be created before the fork; CoW pages of Python objects get
  copied anyway as reference counts change; and `--max-requests` recycling would re-fork from a master that already holds the
  model, hiding real startup cost. Per-worker loading is slower to boot but is simple and safe.
- *Lazy load with a cache but a "warm" background task after start.* The server would accept requests while cold, so the first
  user still pays seconds; the mentor's complaint is exactly that.
- *ONNX Runtime (or quantised int8) for query embedding.* Would cut load time and per-query latency and RAM, but it is a new
  runtime, a model export step in the image build, and a check that vectors match the stored ones closely enough not to shift
  search results (query and chunk embeddings must come from the same model numerics). The cost today is a one-time start-up
  delay, which warm-up removes; revisit only if per-query latency or memory becomes the problem.
- *A single separate embedding service.* One model copy for all workers, but another container, network hop and failure mode for a
  36 MB model.

### D7. MP4s are stored with the index at the front (faststart)

**Problem.** Camera and screen-recorder MP4s have the `moov` index after the media data. A browser must read it before it can play,
so it makes range requests to the end of the file first (extra round trips to S3 before the first frame).

**Decision.**
- At ingest, after the audio is extracted, an MP4/MOV whose `moov` comes after `mdat` is remuxed with
  `ffmpeg -map 0:v -map 0:a? -c copy -movflags +faststart` (stream copy: no re-encode, no quality loss, about 0.1 s for a short clip and
  disk-speed for a long one) and uploaded over the same S3 key, so the database row and any open links are unchanged.
- Detection reads only the top-level box headers (`media.moov_before_mdat`), so files that are already faststart are not touched,
  and webm/mkv (which have no moov atom) are skipped. Reprocessing is therefore idempotent.
- It is best effort: the original plays, only slower to start, so a remux or upload problem is logged as a WARNING with the job id
  and never fails ingestion (the script below can retry it later).
- Existing videos: `python -m app.scripts.remux_faststart` (`--dry-run`, `--video-id`, `--keep-original`). Each file is
  downloaded, remuxed, verified (index first, duration within 1 s, audio still present) and only then uploaded over the original;
  a failure leaves that video as it was and the script continues; exit code 1 if any failed.

**Known limits.** Subtitle and data tracks are not carried over (only all video and audio streams). The replace is in place, so
without `--keep-original` or bucket versioning there is no way back to the original bytes; the verification is the safeguard.
A presigned URL already handed to a browser keeps serving the old object until it expires (1 h).

**Rejected.**
- *Re-encoding to a web-friendly profile.* Slow, lossy, expensive on 2 vCPUs, and not needed for the problem.
- *HLS (or DASH) streaming.* Would give adaptive bitrate and instant start, but needs segmenting at ingest, many small objects per
  video plus playlists, either hls.js (a new frontend dependency) or Safari-only native support, and a presigning scheme for every
  segment. Far more machinery than a one-line remux that fixes the actual delay; worthwhile only for long videos or bad networks.
- *Fixing it on the fly in the API (proxy the video).* Puts video bytes through the 2 vCPU server, which is what S3 pre-signed URLs
  avoid.
- *A new key for the remuxed file and a column pointing to it.* Needs a schema change and ALTER for no benefit over replacing the
  object under the same key.

## 2026-10 Production network review (mentor's browser recording)

The recording showed: 3-4 MB of video downloaded per page on Search and Library, 11 range requests for one video in 6 seconds,
`/api/videos` fetched every 5 s on every page (about 60 of 150 requests), empty results that did not explain themselves, search
getting slow during a transcription, and a 10-result search taking 292 ms against 28-78 ms for empty ones.

### D1. Video bytes are fetched only when the user opens a video

**Decision.** There is exactly one `<video>` in the app, the one in `VideoPlayer`, and it has `preload="none"`. Result rows, cards
and the home grid show a colour tile (`VideoThumb`, no `<video>`, no stream URL request). The stream URL is requested when the user
opens `/watch/:id` (a result click navigates there), and the bytes start only when playback starts. With `?t=` the player sets
`currentTime` first and then calls `play()`, so the part before that second is never downloaded.

**Why a plain tile.** The thumbnails *were* the cost: each one was a hidden `<video preload="metadata" src="...#t=...">`, which makes
the browser fetch the file header and the frames near the start second (several range requests per video, repeated for every result
of the same video at different start times), plus one `/stream` API call per card.

**Rejected.**
- *`preload="none"` on the thumbnail videos plus a `poster`.* Without a downloaded frame there is no picture, so it equals the tile
  with extra elements and extra `/stream` calls for nothing.
- *Fetch the thumbnail frame only on hover.* Still a request for something the user did not ask for, and no hover on phones.
- *Server-generated poster images* (ffmpeg grabs one JPEG at ingest, stored next to the transcript in S3). This is the better
  product: real pictures for ~20 KB each. It needs an ingest step, storage, a backfill for existing videos and a CORS/URL decision,
  so it is a separate feature, not part of this fix. Do it if tiles feel too plain.
- *`preload="metadata"` on the single player.* Costs a request on every watch-page open even if the user never presses play, and the
  recording's complaint was about downloads nobody asked for.

**Consequences.** The watch page shows the video's controls but no frame until play (no poster). `tests/noBackgroundRequests.test.mjs`
fails the build if a second `<video>`, a `<video>` without `preload="none"`, or a stream-URL request from anywhere but the player appears.

### D2. Job progress arrives by Server-Sent Events; nothing polls

**Decision.**
- The worker publishes every job state change (`stage`, `progress`, `error`) to the Redis pub/sub channel `svs:job-events`, right
  after the database commit. The upload and reprocess routes publish the first `queued` state, so the card exists before the worker
  picks the job up. Publishing is fire-and-forget and fail-open: Redis trouble never fails a job or an upload (state is in Postgres).
- `GET /api/jobs/events`: admin only (401 without a token, 403 for a normal user), `text/event-stream`, `Cache-Control: no-cache,
  no-transform`, `X-Accel-Buffering: no`, a `: keep-alive` comment every `SSE_KEEPALIVE_SEC` (15 s). Caddy already runs the whole
  `/api/*` handler with `flush_interval -1` and compresses only the static files, so the stream is unbuffered there.
- On connect it subscribes to the channel first and reads the snapshot second, then sends one `snapshot` event with every active job
  and then one `job` event per change. A refresh therefore shows correct progress immediately, and a change between "subscribe" and
  "read" cannot be lost. If Redis is down before the stream starts the answer is a real 503 (the client retries with backoff).
- The endpoint is `async` with `redis.asyncio`: when the browser disconnects, Starlette cancels the generator and its `finally` closes
  the Redis connection. It is registered before the videos router because `/api/jobs/{job_id}` would otherwise try to parse
  "events" as a UUID.
- Frontend: `lib/jobFeed.ts` (pure: parser, reducer, reconnect loop), `hooks/useJobFeed.ts`. The admin page connects only while
  something is active: a video with status `uploaded`, an upload card still waiting for its first event, or a running job in the feed.
  When the last job finishes the connection is closed. A finished job triggers exactly one reload of `/api/videos`. `useVideos` has no
  timer any more, so Home, Library, Search and Watch load their list once and then make no requests.
- Reconnecting after a drop (1 s, 2 s, 4 s ... 30 s, reset by the next snapshot) is not polling: a healthy connection costs zero
  requests. 401/403/404 stop it for good.
- The 250 ms `useCountdown` timer (repainting a 429 "try again in Ns" label) is now a chained `setTimeout`, so the codebase has no
  `setInterval` at all and that is testable; it never touches the network.

**Rejected.**
- *Longer polling interval (30 s) or polling only on the admin page.* Still polling; still wrong after a refresh until the next tick;
  the mentor asked for none.
- *WebSocket.* Two-way is not needed; SSE works through Caddy as it is, reuses the bearer-token fetch client and the SSE parser
  already used by `/api/ask/stream`, and reconnects by itself.
- *`EventSource`.* Cannot send the `Authorization` header (the access token lives in memory only).
- *Postgres `LISTEN/NOTIFY` instead of Redis pub/sub.* Works, but ties up a database connection per listener and the worker already
  talks to Redis (Celery). Redis pub/sub is simpler.
- *Redis Streams* (history + replay). More machinery than needed: the Postgres `jobs` table is the history, and the snapshot gives
  every new subscriber the current truth.
- *A sync (threadpool) generator.* A disconnect would not cancel it until its next write; with keep-alives every 15 s a dead
  connection could pin a Redis connection and a thread for that long.
- *Leaving `GET /api/jobs/{id}` out.* Kept: scripts (`smoke_ingest.ps1`) and manual checks use it; the web app does not.

**Known limits.** A video whose job died without a final state stays `uploaded`, so the admin page keeps its (silent) connection
open until the video is reprocessed or deleted. Each open stream holds one asyncio task and one Redis connection: fine for a few
admins, not designed for hundreds. A stale `queued` event delivered after the snapshot can show an older percentage for a moment;
the next event corrects it, and a late event can never turn a finished job back into a running one.

### D3. Empty search results explain themselves

**Decision.** `No results for '<query>'`. If any video has status `uploaded` (still processing) the line `N videos are still processing
and are not searchable yet.` is added (`1 video is ... is not`). The count comes from the video list the page already loaded, so there is
no extra request.

**Rejected.** *Asking the server for the count on every empty result* (a request nobody needs most of the time).
*Counting `failed` videos as processing* (they will never become searchable on their own; saying "still processing" would be wrong).
**Limit:** the list is as old as the page; a video that finishes after the page loaded is not reflected until the next visit.

### D4. The worker leaves one CPU for the API

**Decision.** `WHISPER_THREADS` (default 1) is passed to faster-whisper as `cpu_threads` (clamped to at least 1: 0 means "all
cores" to CTranslate2). `TORCH_THREADS` (default 1) is applied with `torch.set_num_threads` when each Celery worker process starts
(`worker_process_init`), which caps the embedding and sentence-embedding work at ingest. Both are runtime settings in `.env.prod`:
no rebuild.

**Expected cost.** faster-whisper defaults to 4 threads, which on the 2 vCPU t3.medium saturates both CPUs. With 1 thread, expect
transcription roughly 1.6-2x slower, and the embedding step at the end a similar factor (it is short compared with transcription).
Using the verified baseline of "28-minute video -> 78 chunks in about 150 s" as a guide, plan for about 4-5 minutes for such a video
on that instance, in exchange for search that stays responsive. This is an estimate from thread scaling, not a measurement on the EC2
instance (see the verification list in the change summary). t3 instances are also burstable: sustained CPU use spends CPU credits
whatever the thread count, which is another reason not to run both CPUs flat out.

**Rejected.**
- *Lowering the worker's OS priority (`nice`) and leaving threads alone.* Helps the scheduler favour the API, but containers cannot
  raise priority back, it does nothing for memory-bandwidth contention, and a hard thread cap gives a predictable result.
- *Pinning containers to CPUs (`cpuset`).* Reserves a core even when no job runs, wasting half the machine when idle.
- *`--concurrency` changes.* It is already 1; the cost is within one job.
- *A smaller Whisper model.* Changes transcript quality, which is a product decision, not a performance fix.

### D5. Search timing is measured per request

**Decision.** `/api/search` returns a standard `Server-Timing` header (`log`, `embed`, `vector_sql`, `keyword_sql`, `hl_lookup`,
`hl_embed`, `rerank` when used, `total`), so the numbers appear in DevTools > Network > Timing next to the request, plus
`X-Search-Highlight: stored=N, fallback=M, single=K` and the existing `X-Search-Ms`. `fallback>0` means highlights were computed by
embedding sentences during the request (the 500 ms path from the Phase 11 notes); it also logs a WARNING that names the fix.
A per-request `Timings` object is passed down as an optional `timings=` argument (None for the agent and the eval harness, no cost).
`python -m app.scripts.check_highlights` reads the database and reports, per video, how many multi-sentence chunks have usable
stored rows (`stored`), have none (`missing`) or have rows from an older splitter (`stale`); exit code 1 if any chunk would fall
back. That is how to confirm on the real server that `chunk_sentences` is in use.

**Why headers and not only logs.** The mentor reads a browser recording; headers show up in it. A debug log line (`search ...`) is
written too for the server side.

**Rejected.** *A tracing library* (OpenTelemetry): new dependency and exporter for a question that headers answer. *Always
timing inside `retrieval` via globals/contextvars:* hides data flow; an explicit optional argument is simpler and testable.

**What the 292 ms could be** (not yet known; the new headers will say): a cold query embedding (~100 ms, cached ~0.5 ms), the
highlight fallback (~500 ms for 10 hits, would show as `hl_embed`), a competing transcription using both CPUs (item D4), or
two SQL statements in hybrid mode. Empty results skip highlights, which is why they are always fast.

## Earlier decisions

See the "Environment decisions" section of `Claude.md` (SeaweedFS instead of MinIO, CPU-only PyTorch, WAV arrays for
faster-whisper, worker failure rules, auth rules, hybrid search, stored sentence vectors, production layout).
