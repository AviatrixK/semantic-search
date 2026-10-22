# CLAUDE.md — Semantic Video Search

## What this is
Semantic search over video. Admin uploads video → Celery worker runs ffmpeg → faster-whisper → chunking
(~30s windows, 5s overlap) → sentence-transformers embeddings → Postgres+pgvector. Users search by meaning
and jump to timestamps. (There is no question-answering / LLM feature: Ask was removed, see the "Ask was removed" bullet.)

## Stack (do not change without asking)
- Backend: Python 3.11, FastAPI, SQLAlchemy 2.0, Pydantic v2, pydantic-settings
- DB: Postgres 16 + pgvector (schema in `db/init.sql`; HNSW cosine index; generated `tsv` column)
- Queue: Celery + Redis. Storage: SeaweedFS (S3 API on :9000 via boto3; compose service is still named `minio`)
- ML: faster-whisper, sentence-transformers `all-MiniLM-L6-v2` (EMBED_DIM=384)
- Frontend: React + Vite + TypeScript, react-router, plain CSS modules (no UI kit unless asked)
- Auth: JWT access token (15 min, kept in memory) + rotating refresh token (httpOnly cookie, hashed in DB)

## Architecture rules
- Routes are thin. Business logic lives in `app/services/`.
- Heavy models load once per process (`lru_cache`), never per request.
- Every new endpoint has auth (`current_user` or `require_admin`) unless explicitly public.
- Config only via `app/core/config.py` + `.env`. Never hardcode secrets or URLs.
- Schema changes: edit `db/init.sql` AND tell me the ALTER statement for existing databases.

## Working rules
- Work one phase at a time from `BuildPhase.md` (repo root). Do not start the next phase unprompted.
- Write or update tests for every service you touch. Run `pytest` before saying you're done.
- Keep diffs focused. Don't refactor unrelated code.
- At the end of each phase: summarize files changed, how to test, and propose the commit message.
- Developer is on Windows (PowerShell) with Python 3.13 locally; Docker uses 3.11. Give PowerShell commands.

## Environment decisions (do not revert)
- **Storage = SeaweedFS, not MinIO.** MinIO removed its public Docker Hub/Quay images (pulls fail with 401).
  Compose service name stays `minio`: `image: chrislusf/seaweedfs`, `command: server -dir=/data -s3 -s3.port=9000`,
  port 9000 only, volume `miniodata:/data`. Same S3 API, so boto3 code and `.env` are unchanged.
  There is NO web console (no port 9001). It accepts any credentials in dev. Its startup logs print harmless errors
  ("Not current leader", "Failed to load IAM configuration").
- **Dockerfiles: CPU-only PyTorch.** Install torch from `https://download.pytorch.org/whl/cpu` in its own layer BEFORE
  `pip install -r requirements.txt`; both pip steps use `--mount=type=cache,target=/root/.cache/pip`. Every
  Dockerfile (including the Phase 13 production ones) keeps this structure. Each build context needs a `.dockerignore`
  (`backend/.dockerignore`: .venv, __pycache__, *.pyc, .pytest_cache, *.wav, *.mp4, *.transcript.json).
- **Transcription never passes a path to faster-whisper.** faster-whisper 1.2.1 calls `av.open(metadata_errors=...)`,
  which av 18+ rejects, and older av has no Windows/3.13 wheel. `transcription._load_wav()` reads the 16 kHz mono
  16-bit WAV with stdlib `wave` into a float32 numpy array and `transcribe()` passes that array. Always pass an array.
  Do NOT pin `av` in requirements.txt. `faster-whisper` stays `>=1.1,<2`.
- **Silent video test case:** `sample/EduSphereDemonstration.mp4` has a fully silent audio track (-91 dB), so Whisper
  returns 0 segments. "No speech detected in this video" must be a clear message in the CLI (Phase 1) and a failed job with a readable
  error in the worker (Phase 2), never an empty transcript or a "done" job with zero chunks. `sample/` is gitignored.
- **Main test file:** `sample/purpose.mp4` (19.5 min, 59 chunks locally with defaults). `scripts/smoke_ingest.ps1` reports the
  chunk count (`-ExpectedChunks 59` to compare). Storage is verified there by fetching the pre-signed stream URL (no console).
- **Worker failure rules:** never retry bad input. `jobs.error` holds only short user-facing text (never paths, tool output or
  memory addresses; the raw output goes to the worker log with the job_id):
  corrupt/unreadable file -> "This file is not a valid video."; no audio stream -> "This video has no audio track.";
  `NoSpeechError` -> "No speech detected in this video.". Each exception class carries its own `user_message`. Unknown errors
  -> a generic message + `log.exception`. Only transient S3/DB errors retry (exponential backoff, max 3). Every failed job logs a
  WARNING with job_id, video_id and the user-facing message.
- **Upload route:** insert the video, `db.flush()`, then add the job (the job FK needs the video row). If the DB write fails after
  the file reached storage, delete the stored object and return a readable 500 body.
- **chunks has `UNIQUE (video_id, idx)`** (constraint `chunks_video_id_idx_key`). Reprocess returns 409 while a job for that video is running.
- **Verified baseline:** a 28-minute video -> 78 chunks in about 150 s on CPU; reprocess keeps the count at 78.
- **psql:** always pass `-P pager=off` (e.g. `docker compose exec postgres psql -U svs -P pager=off -c "..."`).
- **Smoke script** must work on Windows PowerShell 5.1: no `Invoke-WebRequest -Headers @{Range=...}` (use `curl.exe -r`), no
  PS7-only syntax.
- **Auth rules:** passwords need 8+ chars with a letter and a digit (max 72 bytes: bcrypt limit). Refresh tokens are single-use
  (atomic claim); presenting a revoked one revokes ALL of that user's tokens. Missing/invalid access token -> 401 (never 403).
  Rate limits (Redis fixed window, `app/core/ratelimit.py`, fail-open if Redis is down): login 5/min per IP+email, search 30/min
  per user -> 429 + `Retry-After`. Expired refresh tokens are purged hourly by Celery beat (worker runs with `-B`).
- **Tests:** unit tests need no services. Integration tests use the `svs_test` database (rebuilt from `db/init.sql` each session,
  truncated per test) and Redis db 15 (flushed per test); conftest forces both and refuses anything else, so dev data and
  dev limiter state are never touched. Create the DB once on an existing volume:
  `docker compose exec postgres psql -U svs -d postgres -P pager=off -c "CREATE DATABASE svs_test;"`
  (fresh volumes get it from `db/00-create-test-db.sql`). Run: `docker compose exec api pytest`. Inside Docker a missing test DB
  fails loudly; on a bare machine integration tests skip. Tests never load Whisper or embedding models.
- **Search (`/api/search`):** `retrieval.vector_search` over-fetches 3x, drops hits below `MIN_SCORE` (0.25), dedupes overlapping
  hits of the same video (greedy, best score wins; touching ranges are not overlaps), then keeps k. The threshold and dedupe are
  pure functions in `services/search_logic.py` (unit-tested without DB or model). Filters: `video_id`, `uploaded_after` (date, UTC,
  inclusive). Query embeddings are cached in Redis (`emb:` + sha256(provider + model + dimensions + query), TTL 7 days, fail-open). Response header
  `X-Search-Ms` (exposed through CORS). `highlight` = best sentence of the chunk by cosine.
- **Highlights use stored sentence vectors.** Embedding ~80 sentences at query time measured ~525 ms on this CPU (budget 50 ms), so
  the worker embeds each multi-sentence chunk's sentences once at ingest into `chunk_sentences` (`services/sentences.py`); search
  only reads them (~13 ms for k=10). Chunks without usable stored rows fall back to on-the-fly embedding (correct, slow), and a
  missing table degrades the same way. Old data: `python -m app.scripts.backfill_sentences`. Never embed sentences per request
  when stored ones exist. Measured: cold query embedding ~98 ms, cached ~0.5 ms, search without highlight ~4 ms inside Docker.
- **Highlight sentences must be whole sentences.** `split_sentences` splits on . ! ? only before a capital/digit/quote and never after
  abbreviations (Dr., e.g.) or initials; a sentence stays whole up to 80 words (never cut into fixed word windows: that produced
  "world, I've discovered ... how they use their voice, how"). Only longer run-ons are cut, at clause punctuation, and a cut never
  lands after a dangling word (the/to/and/how...). Unfinished fragments at a chunk edge are trimmed of trailing dangling words. A chunk
  that starts mid-sentence still yields a lowercase first fragment: its own text cannot complete it. **Changing the splitter makes
  stored `chunk_sentences` stale (search then falls back to slow on-the-fly embedding): re-run `backfill_sentences`, which rebuilds
  stale rows too, and restart the worker.**
- **Hybrid search (Phase 11).** `/api/search?mode=hybrid|vector|keyword` (default hybrid) uses
  `retrieval.hybrid_search`: top `HYBRID_CANDIDATES` (30) of vector (after the MIN_SCORE cut) and keyword (`tsv @@ websearch_to_tsquery`,
  `ts_rank_cd`, no cosine cut) fused by `search_logic.rrf_merge` (k=`RRF_K`=60), dedupe by `rrf`, optional `services/rerank.py` cross-encoder
  (`RERANK`, default off, model loaded lazily once, failure -> fused order). `score` is always cosine (the UI shows it); keyword-only mode
  uses rank/(1+rank). Keyword terms are
  ANDed, so long natural-language questions often get 0 keyword hits and hybrid then equals vector. No schema change (tsv + GIN already existed).
- **Eval harness (Phase 12, `backend/eval/`, see its README).** Gold file `eval/queries.jsonl` (hand-labelled by the developer: NEVER invent
  gold entries; for pipeline smoke tests build a throwaway file from chunk text under /tmp). Same query text on several lines = one query with
  several gold ranges. `metrics.py` is pure (tested in `tests/test_eval_metrics.py`); `run_eval.py` (vector/keyword/hybrid/hybrid+rerank,
  Recall@1/@5, MRR, median warm latency, silently-falling-back reranker is reported as skipped, not as a result);
  `add_query.py` (interactive labelling). Run inside the api container. On Git Bash set `MSYS_NO_PATHCONV=1` when passing /tmp paths.
- **pgvector HNSW gotcha:** it returns at most `hnsw.ef_search` (default 40) rows regardless of LIMIT and applies WHERE filters after
  the index scan; `vector_search` does `SET LOCAL hnsw.ef_search` to cover its candidate limit.
- **Frontend (`frontend/`, run with `npm run dev` on the host, NOT in Docker; no frontend compose service: node image is a big
  download).** Dependencies are deliberately minimal: react, react-dom, react-router-dom, typescript, vite, plus the type-only
  `@types/react` / `@types/react-dom` that TypeScript needs. Ask before adding anything else (no @vitejs/plugin-react, vitest,
  eslint, axios...). Plain CSS modules. Vite proxies `/api` and `/auth` to the backend (override: `API_TARGET` in
  `frontend/.env.local`), so the refresh cookie (path=/auth) is same-origin: never give frontend routes a path starting with
  `/api` or `/auth`. Access token: memory only (never localStorage). All HTTP goes through `src/api/client.ts`.
- **Refresh safety (client):** a refresh token is single-use and replaying one revokes every session, so there is only ever ONE
  `/auth/refresh` in flight: in-tab single flight (parallel 401s, React StrictMode's double effect) plus a cross-tab Web Locks
  mutex (all tabs share one cookie). A 401 retries once; refresh 401 -> log out; refresh network/5xx errors do NOT log out.
  Auth endpoints never trigger a refresh. 429 -> message from `Retry-After` and a disabled button with a countdown.
- **Frontend tests:** `cd frontend; npm test` (tsc compiles src/api + src/lib, then Node's built-in test runner on `tests/*.test.mjs`;
  no test framework dependency). Logic that must be tested lives outside React components.
- **Admin videos page (`/admin`, `pages/AdminVideos.tsx`).** Uploads use `api.upload` (XMLHttpRequest, for progress events) which shares
  `withAuthRetry` with `api.request`: same 401 -> single-flight refresh -> retry, same ApiError. Uploads go ONE AT A TIME (queue reducer
  in `lib/uploadQueue.ts`); server-side processing of earlier files runs in parallel, each with its own `JobStatus` card fed by the
  live job feed (SSE, see the "No polling" bullet; `JobStatus` is purely presentational). `buildUploadForm` re-wraps the file with the right MIME type: browsers send "" for
  .mkv and the backend answers 415. Client-side size/type checks use `VITE_MAX_UPLOAD_MB` (default 500): keep it equal to the backend's
  `MAX_UPLOAD_MB`. `Video.status` "uploaded" means stored and waiting for / in processing (UI badge "Processing"); the table reloads once
  when the feed reports a job finished. `GET /api/videos` returns `created_at`.
- **Search UX (now `/search`, see the redesign bullet; the combobox/debounce below is the searcher's logic), Library (`/library`), Watch (`/watch/:videoId?t=`).** Search is a debounced (400 ms), cancellable combobox:
  `lib/searcher.ts` (one request per pause, newer queries abort older ones, unchanged queries are not repeated, queries < 2 chars never
  hit the server); focus stays in the box, arrows move `aria-activedescendant`, Enter plays the selected result or searches. The query
  lives in `?q=`. A 429 shows a countdown and retries once automatically. `VideoPlayer` takes a `SeekRequest {videoId, t, play, nonce}`:
  presigned URLs come from `api/streams.ts` (cached per video for 50 min, below the backend's 1 h signature; an expired one is refetched
  once on a media error), another moment of the loaded video only seeks, blocked autoplay shows "Press play". `?t=` accepts 123, 2:03, 1m30s.
- **The transcript must not show chunk overlap.** `/transcript` returns ~30 s chunks that overlap by 4-11 s (the overlap is whole segments,
  so a chunk starts with exactly the text its predecessor ends with). `lib/transcript.ts` trims that repeated start (only when the chunks
  also overlap in time), splits into sentences and interpolates each sentence's start from its position in the text (accurate to a second
  or two). Never render raw chunks as transcript lines. Real chunk text is the test fixture.
- **CSS gotcha:** `position: sticky` on a grid item only moves inside its own grid row. For a stacked (mobile) layout use `display: block`,
  not a one-column grid, or the "sticky" element never sticks.
- **Verification that worked:** besides `npm test`, a headless-Edge run over the DevTools protocol (Node's built-in WebSocket, no new
  dependency) against a throwaway Postgres/Redis/SeaweedFS with the real video and real chunks caught real bugs that unit tests could not.
  Use a different port than 5173 (the developer may have `npm run dev` running) and never kill processes you did not start.
- **Ask was removed (Round 5).** The Ask feature (RAG + hand-rolled Gemini agent, router, fallback models, circuit breaker, chat memory,
  daily token budget, the `/ask` page and `/api/ask*`) is gone on purpose: do not re-add LLM/answer-generation code, `LLM_*`/`AGENT_*`/`RAG_*`/
  `ASK_*`/`CHAT_MEMORY_*`/`DAILY_TOKEN_BUDGET` settings or `google-genai` without being asked. (`GEMINI_API_KEY` exists again, but ONLY for Gemini
  EMBEDDINGS, see the "Hosted models" bullet; it is not an LLM feature.) KEPT on purpose: `GET /api/jobs/events` (job-progress
  SSE) with `SSE_KEEPALIVE_SEC`, `lib/sse.ts` and `api.stream` (now used only by the job feed), the hybrid search and its eval harness (the
  `multi` gold query type stays: a question needing several moments), and `search_logs` (its `mode` column may hold old `'ask'` rows: leave
  them). Ask left no tables of its own; its Redis keys (`chat:*`, `tokens:*`, `llm:cb:*`, `rl:ask:*`) simply expire.
- **No polling, no preloading (mentor review of a production network recording).** The frontend has NO `setInterval` and loads NO video bytes
  until the user opens one (both enforced by `tests/noBackgroundRequests.test.mjs`: no `setInterval`, every `<video>` has `preload="none"`, the
  player is the only `<video>`, only the player asks for stream URLs). Search/Home/Library/Watch make zero requests after their first load.
  Job progress: worker (`_set`) and upload/reprocess routes publish `{id, video_id, stage, progress, error}` to Redis channel `svs:job-events`
  (`services/jobevents.py`, fail-open); `GET /api/jobs/events` (`api/routes/jobs.py`, admin only, async + `redis.asyncio`, registered BEFORE
  the videos router so `/api/jobs/{job_id}` does not swallow it) subscribes first, then sends a `snapshot` of all active jobs, then `job`
  events and `: keep-alive` comments; Redis down before the stream -> real 503. Client: `lib/jobFeed.ts` (parse, reducer, `connectJobFeed`
  reconnects with 1 s..30 s backoff only if the stream drops; 401/403/404 stop for good), `hooks/useJobFeed.ts` (connected only while
  `enabled`); `AdminVideos` enables it while a video is "uploaded", a card awaits its first event, or the feed has a running job. The player
  starts playback itself (`preload="none"` never fires loadedmetadata before play): `startPlayback` sets currentTime first, then `play()`.
  Search empty state: `lib/emptyResults.ts` ("No results for 'q'" + "N videos are still processing and are not searchable yet.").
  A thumbnail from the video file is NOT coming back unless it is a small server-made poster image.
- **Warm start and faststart (Round 1 review).** The API loads the embedding model and encodes once in the FastAPI lifespan (`embedding.warm_up`,
  `WARMUP_ON_START`, default true), in every gunicorn worker; `/health` answers 503 until it is done (`app.state.ready`), uvicorn takes no
  request before lifespan startup ends, and a failed warm-up kills the worker. Measured: first query after warm-up 14 ms vs 11.7 s cold.
  The thread limit is the Celery worker's only (`worker_process_init`); a test fails if API code calls `limit_torch_threads`. Do NOT use
  gunicorn `--preload` (see docs/DECISIONS.md). Ingest remuxes MP4/MOV with `-c copy -movflags +faststart` (`media.remux_faststart`,
  `tasks._store_faststart`, best effort, replaces the stored object under the same key); `media.moov_before_mdat` reads the box headers to
  decide. Existing videos: `python -m app.scripts.remux_faststart [--dry-run] [--keep-original] [--video-id ID]`: decides "already fast" from a ranged
  read of the first 64 KB (no download), skips videos with a running job, verifies duration/audio before replacing; re-runs change nothing.
  webm/mkv have no moov atom: untouched. `/favicon.ico` + `/favicon.svg` live in `frontend/public/` (regenerate the .ico with
  `node frontend/scripts/make-favicon.mjs`). Dev compose runs uvicorn with `--timeout-graceful-shutdown 3` because SSE streams otherwise
  block `--reload` for as long as an admin tab is open.
- **Logging, duplicate uploads, shortcuts (Round 3).** `app/core/logging.py::setup_logging()` (idempotent; stdout; format `time LEVEL
  logger message`; level from `LOG_LEVEL`, default INFO; botocore/urllib3/httpx stay WARNING unless DEBUG) is called first in the API
  lifespan and from Celery's `setup_logging` signal (covers worker AND beat; `--loglevel` is then ignored). Under gunicorn the root logger
  is WARNING, so without it every `log.info` was lost. Access logs are not duplicated: uvicorn/gunicorn loggers keep their own handlers
  and `propagate=False`. **Duplicate uploads:** `videos.sha256` (nullable, unique index `videos_sha256_key`; init.sql + `db/migrations/
  2026-10-12-videos-sha256.sql`), hashed before anything is stored (`services/dedupe.py`); a repeat is `409` with `{detail, video_id,
  title, status}`, also on the race where two identical uploads pass the check together (IntegrityError -> object deleted -> 409).
  `python -m app.scripts.backfill_sha256 [--dry-run]` fills old videos, prefers `originals/<key>` (the remux backup), reports duplicates
  without deleting. **Order on a server: migrate, backfill, THEN `remux_faststart`** (remuxing rewrites the bytes, so a later hash would
  not match the file an admin re-uploads). Prod compose rotates logs for all six services (json-file 10m x 3; tested on the resolved
  config). `Makefile` (`make ps | logs [s=worker] | deploy | shell-api`) wraps `docker compose --env-file .env.prod -f
  docker-compose.prod.yml`.
- **Hosted models (Round 6).** Defaults STAY local (`TRANSCRIBER=whisper`, `EMBED_PROVIDER=local`); the local ML packages stay in the images
  until hosted defaults are chosen from `eval/run_provider_eval.py`. **Transcription:** `services/sarvam.py` (Sarvam batch API, httpx, no SDK;
  REST flow copied from the official SDK source) returns sentence-level timestamps. **ONE speech-to-text job per video** (mode `transcribe`,
  native script, INR 30/h); `text_roman` / `text_en` of each CHUNK are made from its native text by Sarvam's text APIs (`POST /transliterate`
  and `POST /translate`, target en-IN, INR 20 per 10K characters each, `sarvam.add_text_variants`; `SARVAM_TEXT_VARIANTS` = `roman,en` picks
  which; `SARVAM_TRANSLATE_MODEL` sarvam-translate:v1 = 23 languages / 2000 chars per request, mayura:v1 = 11 / 1000; /transliterate has 11
  languages and no documented size limit, so `SARVAM_TRANSLIT_MAX_CHARS`=1000; texts over a limit are split at sentence ends; en-IN speech
  needs no calls; a 422 on one text leaves that value empty). The old three-audio-jobs-aligned-by-time design (`align_text`) is GONE. Cost:
  about INR 342 per hour of video (30 + 2 x 78,000 chars/h x 20/10K; the 78,000 is measured on the dev library and includes the 5 s chunk
  overlap), the text APIs are 90% of it. **Idempotent:** `videos.sarvam_job_id` / `sarvam_job_status` are committed the moment the job exists
  and after each step (created, uploaded, started, then Sarvam's states, finally `consumed` when the chunks are stored or the failure is
  permanent); a Celery retry hands them back to `transcribe_audio`, which resumes the same job (re-upload only if `created`, start only if
  Sarvam still says `accepted`, else just poll/download) and starts a new one only if the saved job `failed`, is unknown (404), or is
  `consumed`. Text-API calls are NOT resumable: a retry after the transcript step pays for them again. **Models:** `SARVAM_MODEL=saaras:v4`
  works; languages in `SARVAM_V4_UNSUPPORTED` (empty: the docs list the same 23 languages for v4 and v3) and any job request v4 answers
  with 400/422 are retried on `SARVAM_FALLBACK_MODEL` (v3). Audio over `SARVAM_MAX_FILE_SEC` (batch limit 2 h) is split and offset within
  the same job. Chunks store
  `text` (= `text_native`), `text_roman`, `text_en`; `videos.language_hint` (upload dropdown, `services/languages.py` <-> `frontend/src/lib/
  languages.ts`, test keeps them equal) and `videos.transcriber`. **Embeddings:** `services/embedders.py` (one interface; separate
  `embed_documents` / `query_vector` because task types differ: gemini-embedding-2 puts the task in the TEXT, -001 uses task_type, cohere
  input_type, openai symmetric); hosted vectors go to `chunk_embeddings` (dimensionless `vector` column + one partial HNSW expression index
  per `provider/model@dims`, `services/vectorstore.py`; search takes the better of native/en per chunk); `chunks.embedding vector(384)` is the
  LOCAL model's and is nullable. `EMBED_DIMENSIONS` default 1536 (HNSW max 2000, verified). **Resilience:** query embedding 2 s timeout + 1
  retry, failure -> keyword-only search (WARNING log, `X-Search-Fallback: keyword-only`), cache key = provider+model+dims+query, 7 days.
  Hosted providers get LEXICAL highlights (`best_lexical_index`), no sentence vectors. **Keyword search = TWO indexes** in one query (`retrieval._keyword_candidates`
  ORs them, rank = the two `ts_rank_cd` added, so a chunk found by both ranks higher; still ONE keyword list in the RRF merge): `chunks.tsv`
  = `'simple'` over native + romanised text, `chunks.tsv_en` = `'english'` (stemming) over `coalesce(text_en, text)` (Whisper English videos
  have no text_en: their text feeds it). Measured on the real library (old English / 3-variant simple / new two indexes): "voices"
  22 / 2 / 22, "speaking" 16 / 4 / 16, "robot" 29 / 15 / 29. **Per-provider `MIN_SCORE_LOCAL|GEMINI|COHERE|OPENAI`** (empty = `MIN_SCORE`,
  `settings.min_score_for(provider)`); hosted values are NOT measured yet. **Token counts:** `HostedEmbedder._count_tokens` adds the
  provider's reported count (Gemini `usageMetadata.promptTokenCount`, Cohere `billed_units`, OpenAI `usage`) and only estimates (4
  chars/token) when the reply has none; `tokens_estimated` says how much was guessed. The UI shows "Showing keyword matches only, as smart
  search is temporarily unavailable." above results when `X-Search-Fallback` is set (`lib/searchFallback.ts`). Startup
  fails if a hosted provider's key is missing; no warm-up for hosted. The API+worker import and start with torch/sentence-transformers/
  faster-whisper blocked (tested) so the images can drop them later. Scripts: `python -m app.scripts.reprocess [--video-ids] [--transcriber]
  [--embed-provider] [--dry-run] [--yes]` (resumable from DB state, asks before paid work), `eval/run_provider_eval.py`. Prices and sources:
  `app/services/pricing.py` (Sarvam STT INR 30/h once + INR 20/10K chars per text version; embeddings per 1M tokens). Migrations
  `db/migrations/2026-10-18-hosted-models.sql` and `2026-10-20-sarvam-single-job.sql`.
  Not verified against the real services: no Sarvam/Gemini/Cohere/OpenAI call has been made (mocked HTTP only); in particular the
  text APIs' real size limits, whether Sarvam accepts a repeated `upload-files` for a `created` job, and whether Gemini returns usage.
- **Worker threads and search timing.** `WHISPER_THREADS` (default 1, faster-whisper `cpu_threads`) and `TORCH_THREADS` (default 1,
  `torch.set_num_threads` in `worker_process_init`) keep one of the server's 2 CPUs free for the API; raise both on a bigger machine.
  `/api/search` sends `Server-Timing` (log, embed, vector_sql, keyword_sql, hl_lookup, hl_embed, rerank, total; visible in DevTools >
  Network > Timing) and `X-Search-Highlight: stored=N, fallback=M, single=K` (`services/timing.py`, `timings=` kwarg of the retrieval
  functions). `fallback>0` means highlights were embedded per request (slow) and logs a WARNING. `python -m app.scripts.check_highlights`
  reports chunk_sentences coverage (exit 1 if any chunk would fall back).
- **Frontend redesign (YouTube-style, LIGHT by default).** Shell: sticky `Header` (menu button, logo, search box, avatar `AccountMenu`) + `Sidebar`
  (full / icon rail via the hamburger, remembered in `svs.sidebar`; a bottom bar on phones) + page. Routes: `/` Home (thumbnail grid + sort chips;
  an old `/?q=` link redirects), `/search?q&mode&video&after` (YouTube-style result rows; the header box is the only search input, Enter submits,
  `/` focuses it everywhere), `/watch/:id?t=` (big player + transcript panel), `/library`, `/admin`. Thumbnails are colour tiles:
  `VideoThumb` has NO `<video>` and makes no request (see the "No polling" bullet). Theme: light by default, dark only by choice (`lib/theme.ts`, `svs.theme`, applied by an inline script in `index.html`); the OS dark
  setting is IGNORED on purpose; dark tokens are the `[data-theme='dark']` block in `index.css`. NO background doodles/patterns (removed on request): plain `--bg`. The product is for education: the logo is a
  graduation cap (no YouTube-like logo or red play button anywhere); the brand colour is indigo `--brand`. Search state: `lib/searchParams.ts` (validated URL <-> filters), `createSearcher` takes filters
  (`setFilters` searches at once; `filtersKey` drops stale responses). A11y/speed: skip link, page title + live region per route (`lib/titles.ts`), Watch/Admin
  are `React.lazy` with navbar prefetch and `RouteErrorBoundary`, skeletons, reduced motion, 44px targets on coarse pointers. Verified with headless
  Edge against a throwaway mock backend serving a real sample video.
- **Production (Phase 13, one EC2 t3.medium, see `docs/DEPLOY.md`).** `docker-compose.prod.yml` (+ `.env.prod`, `Caddyfile`, `backend/Dockerfile.prod`,
  `frontend/Dockerfile`, `scripts/deploy.sh`, `scripts/backup.sh`); run with `--env-file .env.prod`. Services: postgres, redis, api (gunicorn + uvicorn
  workers, `WEB_CONCURRENCY` 2), worker (`--concurrency=1`, no `-B`), beat, caddy (the only published ports 80/443; serves the built React app, proxies
  `/api` and `/auth`, `flush_interval -1` for SSE). Storage is AWS S3: `S3_ENDPOINT` unset = regional AWS endpoint + SigV4 + the EC2 instance role
  (keys unset); set = SeaweedFS as in dev (dev `.env` needs `AWS_REGION=us-east-1`). `APP_ENV=production` (set by the prod compose) makes `Settings` refuse
  default JWT_SECRET / DB password / `COOKIE_SECURE=false` and makes `storage.ensure_bucket` fail instead of creating the bucket. The prod image bakes
  Whisper + embedder (build args `WHISPER_SIZE`, `EMBED_MODEL`). The frontend only plays S3 videos via `<video>` (no CORS rule needed; adding fetch/XHR/crossOrigin
  on S3 URLs would need one). Not verified on a real server yet.
- **Dev machine:** Windows + PowerShell, Python 3.13 in `backend\.venv`, Docker Desktop (WSL 2), ffmpeg on PATH. The project
  path contains spaces: quote every path in commands and scripts. Internet is slow/unreliable: avoid forcing large
  re-downloads and ask before adding a heavy dependency.

## Commands
- Stack: `docker compose up --build`
- Admin: `docker compose exec api python -m app.scripts.create_admin <email> <password>`
- Tests: `docker compose exec api pytest` (all) or `cd backend; pytest` (unit tests; integration skip without svs_test)
- Frontend: `cd frontend; npm run dev` (http://localhost:5173)