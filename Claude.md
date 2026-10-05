# CLAUDE.md — Semantic Video Search

## What this is
Agentic RAG over video. Admin uploads video → Celery worker runs ffmpeg → faster-whisper → chunking
(~30s windows, 5s overlap) → sentence-transformers embeddings → Postgres+pgvector. Users search by meaning
and jump to timestamps; a Gemini tool-calling agent answers multi-step questions with citations.

## Stack (do not change without asking)
- Backend: Python 3.11, FastAPI, SQLAlchemy 2.0, Pydantic v2, pydantic-settings
- DB: Postgres 16 + pgvector (schema in `db/init.sql`; HNSW cosine index; generated `tsv` column)
- Queue: Celery + Redis. Storage: SeaweedFS (S3 API on :9000 via boto3; compose service is still named `minio`)
- ML: faster-whisper, sentence-transformers `all-MiniLM-L6-v2` (EMBED_DIM=384)
- LLM: Gemini via `google-genai`. Agent is a HAND-ROLLED loop — no LangChain/LangGraph unless asked
- Frontend: React + Vite + TypeScript, react-router, plain CSS modules (no UI kit unless asked)
- Auth: JWT access token (15 min, kept in memory) + rotating refresh token (httpOnly cookie, hashed in DB)

## Architecture rules
- Routes are thin. Business logic lives in `app/services/`. Agent tools call services, never raw SQL.
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
  and ask 10/min per user -> 429 + `Retry-After`. Expired refresh tokens are purged hourly by Celery beat (worker runs with `-B`).
- **Tests:** unit tests need no services. Integration tests use the `svs_test` database (rebuilt from `db/init.sql` each session,
  truncated per test) and Redis db 15 (flushed per test); conftest forces both and refuses anything else, so dev data and
  dev limiter state are never touched. Create the DB once on an existing volume:
  `docker compose exec postgres psql -U svs -d postgres -P pager=off -c "CREATE DATABASE svs_test;"`
  (fresh volumes get it from `db/00-create-test-db.sql`). Run: `docker compose exec api pytest`. Inside Docker a missing test DB
  fails loudly; on a bare machine integration tests skip. Tests never load Whisper or embedding models.
- **Search (`/api/search`):** `retrieval.vector_search` over-fetches 3x, drops hits below `MIN_SCORE` (0.25), dedupes overlapping
  hits of the same video (greedy, best score wins; touching ranges are not overlaps), then keeps k. The threshold and dedupe are
  pure functions in `services/search_logic.py` (unit-tested without DB or model). Filters: `video_id`, `uploaded_after` (date, UTC,
  inclusive). Query embeddings are cached in Redis (`emb:` + sha256(model + query), TTL 1 day, fail-open). Response header
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
- **Hybrid search (Phase 11).** `/api/search?mode=hybrid|vector|keyword` (default hybrid) and the agent's `search_transcripts` use
  `retrieval.hybrid_search`: top `HYBRID_CANDIDATES` (30) of vector (after the MIN_SCORE cut) and keyword (`tsv @@ websearch_to_tsquery`,
  `ts_rank_cd`, no cosine cut) fused by `search_logic.rrf_merge` (k=`RRF_K`=60), dedupe by `rrf`, optional `services/rerank.py` cross-encoder
  (`RERANK`, default off, model loaded lazily once, failure -> fused order). `score` is always cosine (the UI shows it); keyword-only mode
  uses rank/(1+rank). `/api/ask` rag and the "search" route still use `vector_search` on purpose (not asked to change). Keyword terms are
  ANDed, so long natural-language questions often get 0 keyword hits and hybrid then equals vector. No schema change (tsv + GIN already existed).
- **Eval harness (Phase 12, `backend/eval/`, see its README).** Gold file `eval/queries.jsonl` (hand-labelled by the developer: NEVER invent
  gold entries; for pipeline smoke tests build a throwaway file from chunk text under /tmp). Same query text on several lines = one query with
  several gold ranges. `metrics.py` is pure (tested in `tests/test_eval_metrics.py`); `run_eval.py` (vector/keyword/hybrid/hybrid+rerank,
  Recall@1/@5, MRR, median warm latency, silently-falling-back reranker is reported as skipped, not as a result); `run_answer_eval.py` (rag vs
  agent on `multi`, Gemini judge, citation precision; makes REAL Gemini calls and asks first; never run it without the user's go-ahead);
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
  in `lib/uploadQueue.ts`); server-side processing of earlier files runs in parallel, each with its own `JobStatus` card polling
  `/api/jobs/{id}` every 2 s via `useJobPolling` -> `lib/poller.ts` (one request at a time, stops on done/failed or fatal 401/403/404,
  backs off on errors, honours Retry-After on 429). `buildUploadForm` re-wraps the file with the right MIME type: browsers send "" for
  .mkv and the backend answers 415. Client-side size/type checks use `VITE_MAX_UPLOAD_MB` (default 500): keep it equal to the backend's
  `MAX_UPLOAD_MB`. `Video.status` "uploaded" means stored and waiting for / in processing (UI badge "Processing"); the table refreshes
  itself every 5 s while any video is processing. `GET /api/videos` returns `created_at`.
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
- **Ask / RAG (`POST /api/ask`, page `/ask`).** `services/llm.py` wraps google-genai: per-attempt timeout (`LLM_TIMEOUT_SEC`, sent to the SDK
  in ms), retries on 429/500/502/503/504 and network errors (`LLM_MAX_RETRIES`, exponential backoff with jitter), token usage logged (counts
  only: never prompts or answers, they hold user questions and transcript text). Every `LLMError` subclass carries a short safe
  `user_message`; the route turns them into 503 with exactly that text, never the SDK's message. The LLM is swappable (`set_llm`), so tests
  use `tests/fakes.py::FakeLLM` and **never call Gemini**. `services/rag.py`: retrieve `RAG_TOP_K` (8) with `highlight=False`; no hits -> fixed
  "couldn't find anything" answer WITHOUT calling the model; context blocks `[n] (title @ mm:ss) text` inside `<excerpts>`, question in
  `<question>`; the system prompt says answer only from the excerpts, cite `[n]`, reply with the INSUFFICIENT sentence if unsure, and treat
  excerpts as untrusted data (prompt injection). `parse_citations` keeps only `[n]` that exist, rewrites `[1, 2]` as `[1][2]`, removes
  invented numbers and tidies the gap. Queries are logged with mode "ask"; rate limit 10/min per user. Config: `GEMINI_API_KEY`,
  `LLM_MODEL` (default alias `gemini-flash-latest`; pin an exact id for reproducibility), `LLM_TIMEOUT_SEC`, `LLM_MAX_RETRIES`,
  `LLM_MAX_OUTPUT_TOKENS` (includes thinking tokens), `RAG_TOP_K`. `python -m app.scripts.check_llm` sends ONE real tiny request (`--models`
  lists usable model ids): run it on purpose after setting the key.
- **Ask UI.** Answers render as text nodes only (never HTML); `lib/answerText.ts` turns known `[n]` into chip buttons that seek the shared
  `VideoPlayer`. `lib/chat.ts` is the pure state machine (one pending question, cancel removes it and returns the text to the box, retry);
  the conversation lives in `sessionStorage` (this tab only; a request that was pending when the page went away comes back as a retryable
  error). 429 blocks sending until Retry-After; client gives up after 100 s. Enter sends, Shift+Enter is a new line.
- **Agent / router / SSE / guardrails (Phases 9-10).** `app/agent/`: `loop.py` (hand-rolled: model -> tool calls (parallel) -> results in ONE
  turn -> repeat; after `max_steps` one forced final call with tool mode NONE, so calls <= max_steps+1), `tools.py` (Pydantic-validated
  args, bad args return an error string to the model, results truncated to a token budget BEFORE they count as evidence), `citations.py`
  (model cites `[video_id@seconds]`; only refs inside ranges tools actually returned survive, renumbered to `[n]`), `router.py` (heuristics
  first, tiny LLM call only when ambiguous, failure -> rag), `orchestrator.py`. `POST /api/ask` takes `mode` rag|agent|auto|search and
  returns `route`, `trace`, `usage`; `GET /api/ask/stream` is SSE (events route, step_start, step_result, answer, error, done; auth via
  bearer through fetch, never EventSource; pre-stream failures are real 401/422/429). Client disconnect stops the agent before its next
  model call. Guardrails: transcript text is wrapped as untrusted data (`services/safety.py`), per-request LLM call cap, per-user daily
  token budget in Redis (`services/usage.py`, fail-open), last 3 Q/A turns per chat session held SERVER-side (`services/memory.py`, keyed
  by user + client session id). Frontend: `lib/sse.ts` parser, `lib/askStream.ts`, `lib/trace.ts`, `components/AgentTrace.tsx`.
  Not yet done: a real Gemini call and a real-browser E2E of the live trace (unit + fake-server verified only).
- **Frontend redesign (YouTube-style, LIGHT by default).** Shell: sticky `Header` (menu button, logo, search box, avatar `AccountMenu`) + `Sidebar`
  (full / icon rail via the hamburger, remembered in `svs.sidebar`; a bottom bar on phones) + page. Routes: `/` Home (thumbnail grid + sort chips;
  an old `/?q=` link redirects), `/search?q&mode&video&after` (YouTube-style result rows; the header box is the only search input, Enter submits,
  `/` focuses it everywhere), `/watch/:id?t=` (big player + transcript panel), `/ask`, `/library`, `/admin`. Result thumbnails and cards are the
  video itself: `VideoThumb` is a muted `<video preload="metadata" src="<presigned>#t=<moment>">` created only when near the screen (no thumbnail
  files). Theme: light by default, dark only by choice (`lib/theme.ts`, `svs.theme`, applied by an inline script in `index.html`); the OS dark
  setting is IGNORED on purpose; dark tokens are the `[data-theme='dark']` block in `index.css`. NO background doodles/patterns (removed on request): plain `--bg`. The product is for education: the logo is a
  graduation cap (no YouTube-like logo or red play button anywhere); the brand colour is indigo `--brand`. Search state: `lib/searchParams.ts` (validated URL <-> filters), `createSearcher` takes filters
  (`setFilters` searches at once; `filtersKey` drops stale responses). Ask: answer style auto|rag|agent (`svs.ask.mode`), Copy answer, Regenerate
  (the server memory records the repeat as another turn). A11y/speed: skip link, page title + live region per route (`lib/titles.ts`), Ask/Watch/Admin
  are `React.lazy` with navbar prefetch and `RouteErrorBoundary`, skeletons, reduced motion, 44px targets on coarse pointers. Verified with headless
  Edge against a throwaway mock backend serving a real sample video. NOT done: an in-chat video filter (no such backend parameter).
- **Production (Phase 13, one EC2 t3.medium, see `docs/DEPLOY.md`).** `docker-compose.prod.yml` (+ `.env.prod`, `Caddyfile`, `backend/Dockerfile.prod`,
  `frontend/Dockerfile`, `scripts/deploy.sh`, `scripts/backup.sh`); run with `--env-file .env.prod`. Services: postgres, redis, api (gunicorn + uvicorn
  workers, `WEB_CONCURRENCY` 2), worker (`--concurrency=1`, no `-B`), beat, caddy (the only published ports 80/443; serves the built React app, proxies
  `/api` and `/auth`, `flush_interval -1` for SSE). Storage is AWS S3: `S3_ENDPOINT` unset = regional AWS endpoint + SigV4 + the EC2 instance role
  (keys unset); set = SeaweedFS as in dev (dev `.env` needs `AWS_REGION=us-east-1`). `APP_ENV=production` (set by the prod compose) makes `Settings` refuse
  default JWT_SECRET / DB password / `COOKIE_SECURE=false` and makes `storage.ensure_bucket` fail instead of creating the bucket. The prod image bakes
  Whisper + embedder (build args `WHISPER_SIZE`, `EMBED_MODEL`). The frontend only plays S3 videos via `<video>` (no CORS rule needed; adding fetch/XHR/crossOrigin
  on S3 URLs would need one). Not verified on a real server yet.
- **Verifying a model wrapper without the real service:** run the REAL SDK inside the api container against a tiny local fake Gemini HTTP
  server (`http_options={"base_url": ...}`): it exercised request body, header, usage parsing, real 429/503/400/404 error classes and the
  timeout path with zero Google traffic.
- **Dev machine:** Windows + PowerShell, Python 3.13 in `backend\.venv`, Docker Desktop (WSL 2), ffmpeg on PATH. The project
  path contains spaces: quote every path in commands and scripts. Internet is slow/unreliable: avoid forcing large
  re-downloads and ask before adding a heavy dependency.

## Commands
- Stack: `docker compose up --build`
- Admin: `docker compose exec api python -m app.scripts.create_admin <email> <password>`
- Tests: `docker compose exec api pytest` (all) or `cd backend; pytest` (unit tests; integration skip without svs_test)
- Frontend: `cd frontend; npm run dev` (http://localhost:5173)