# Semantic Video Search

Upload videos → Whisper transcribes → chunks are embedded into pgvector → users search by meaning
and jump to the exact timestamp. An LLM agent (weeks 7–9) plans multi-step searches and answers with citations.

## Services (docker compose)
| Service | Port | Purpose |
|---|---|---|
| api | 8000 | FastAPI — auth, upload, search (docs at /docs) |
| worker | — | Celery — ffmpeg → Whisper → chunk → embed → store |
| postgres | 5432 | Postgres 16 + pgvector (schema in `db/init.sql`) |
| redis | 6379 | Celery broker |
| minio | 9000 | S3-compatible video storage. Runs **SeaweedFS** (MinIO's public images were pulled); service name kept as `minio`. No web console. |

## Week 1 — transcribe one video (no Docker needed)
```bash
pip install faster-whisper pydantic-settings numpy      # plus ffmpeg on your PATH
python scripts/transcribe.py "path\to\my_video.mp4" --chunks
```

## Run the full stack
```bash
cp .env.example .env            # then set JWT_SECRET
docker compose up --build
docker compose exec api python -m app.scripts.create_admin admin@example.com 'ChangeMe123!'
```
First upload is slow: the worker downloads the Whisper and embedding models once.
The backend image installs CPU-only PyTorch (the default Linux wheel pulls ~3 GB of CUDA libraries) and uses a
`backend/.dockerignore` to keep the build context small. SeaweedFS logs a few harmless errors at startup
("Not current leader", "Failed to load IAM configuration").

> The project path contains spaces. In PowerShell always quote paths: `cd "C:\IMP\Career\WEB DEVELOPMENT\Semantic Video Search"`.

## Notes
- Transcription passes faster-whisper a numpy array (read from the ffmpeg WAV), never a file path, to avoid a PyAV incompatibility. Do not pin `av`.
- `sample/` is gitignored. `sample/EduSphereDemonstration.mp4` has a silent audio track and is the "no speech" test case.

## Try it (Swagger at http://localhost:8000/docs, or curl)
```bash
# login as admin -> copy access_token
curl -s -X POST localhost:8000/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"ChangeMe123!"}'
TOKEN=...
curl -X POST localhost:8000/api/videos -H "Authorization: Bearer $TOKEN" -F "file=@my_video.mp4;type=video/mp4"
curl localhost:8000/api/jobs/<job_id> -H "Authorization: Bearer $TOKEN"        # poll until stage=done
curl "localhost:8000/api/search?q=how+do+interest+rates+work" -H "Authorization: Bearer $TOKEN"
```

## Frontend (React + Vite + TypeScript)
Runs on your machine, not in Docker. With the backend up (`docker compose up`):
```powershell
cd "C:\IMP\Career\WEB DEVELOPMENT\Semantic Video Search\frontend"
npm install        # first time only
npm run dev        # http://localhost:5173
```
Vite proxies `/api` and `/auth` to `http://localhost:8000` (set `API_TARGET` in `frontend/.env.local` to change it), so the
refresh-token cookie is same-origin. Pages: Log in, Register, Home (placeholder), Admin (placeholder, admins only), 404.
`npm run typecheck`, `npm run build`, `npm test` (client logic: refresh single-flight, 429 handling, two-tab safety, upload queue,
job polling, formatting).

The app is laid out like a video site, in a plain light theme, and is meant for learning. **Home** (`/`) is a grid of the videos (thumbnails are real frames
from the video) with Newest / Oldest / Longest / Shortest chips. Type in the **search box in the header** and press Enter (press `/` anywhere to jump to
it): **results** (`/search?q=...`) are listed like YouTube search results, each with the frame where the match starts, the matching sentence in bold and a
"Jump to 4:43" chip; clicking one opens the video at that moment. Above the results pick **Best match / Meaning / Exact words** and filter by video or
upload date (all kept in the address, e.g. `/search?q=pauses&mode=keyword`). The **watch page** (`/watch/<id>?t=123`) has a big player with the
transcript beside it: click a line to jump there, the spoken line is highlighted and followed (scroll by hand to pause that, "Follow playback"
resumes it). `t` accepts seconds, `2:03` or `1m30s`, and "Copy link to this moment" builds such a link. The browser plays the video straight from storage
through a presigned URL, so `S3_PUBLIC_ENDPOINT` must be reachable from the browser (it is `http://localhost:9000` by default). The round avatar (top right)
opens a menu with the dark-mode switch and Log out; the hamburger collapses the left menu. **Ask** has an answer style (Automatic, Quick, Research), and
each answer can be copied or regenerated. **Library** (`/library`) lists the ready videos.

**Admin page** (`/admin`, admins only): drag-and-drop or pick several videos, edit titles, upload them one after another with a progress
bar, and watch each one's processing live (stage and progress, polled every 2 seconds, error text on failure). The library table shows
title, duration (mm:ss), status, created date, and Reprocess / Delete (with confirmation). Friendly messages for 413 (too large),
415 (unsupported type) and 429. If you change the backend's `MAX_UPLOAD_MB`, set the same value as `VITE_MAX_UPLOAD_MB` in
`frontend/.env.local` so the browser rejects oversized files before uploading them.
Create an admin to see the Admin link: `docker compose exec api python -m app.scripts.create_admin you@example.com 'YourPass123'`.

## Search
`GET /api/search?q=...` (login required, 30/min per user). Optional: `k` (1-50, default 10), `video_id`, `uploaded_after=YYYY-MM-DD`
(videos uploaded on/after that day, UTC), `highlight=false`. Hits below `MIN_SCORE` (default 0.25) are dropped, overlapping chunks of
the same video are collapsed to the best one, and each hit carries `highlight`, the sentence of the chunk that best matches the query.
`mode=hybrid` (default) | `vector` | `keyword`: hybrid runs semantic and full-text search (top 30 each, `HYBRID_CANDIDATES`) and merges them
with Reciprocal Rank Fusion (`score = sum 1/(60 + rank)`, `RRF_K`), so exact names/acronyms and paraphrases both work. `score` stays the cosine
similarity; hybrid hits also carry `rrf` and `found_by`. Keyword mode uses `websearch_to_tsquery` (plain words are ANDed, `"phrases"`, `OR`, `-exclude`
work) and needs no model. Optional cross-encoder rerank of the top 20: `RERANK=true` in `.env` (downloads `cross-encoder/ms-marco-MiniLM-L-6-v2`
on first use, adds latency; falls back to the fused order if it cannot load). The agent's `search_transcripts` tool uses hybrid.
The `X-Search-Ms` response header reports server time. Query embeddings are cached in Redis for a day, so a repeated query skips the model.

Sentence vectors for highlights are computed at ingest. For videos ingested before Phase 4, one-time (existing database):
```powershell
docker compose exec postgres psql -U svs -P pager=off -c "CREATE TABLE IF NOT EXISTS chunk_sentences (id UUID PRIMARY KEY, chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE, idx INT NOT NULL, text TEXT NOT NULL, embedding VECTOR(384) NOT NULL, CONSTRAINT chunk_sentences_chunk_id_idx_key UNIQUE (chunk_id, idx));"
docker compose restart worker
docker compose exec api python -m app.scripts.backfill_sentences
```
Until the backfill runs, highlights still work but are computed per request (about half a second).
Re-run the backfill whenever the sentence splitter changes: it rebuilds stale rows as well as filling in missing ones.

## Evaluation
`backend/eval/` measures search and answer quality against a hand-labelled gold set (Recall@1/@5, MRR, latency; LLM-judged groundedness and
relevance, citation precision). See `backend/eval/README.md`: `docker compose exec api python eval/add_query.py`, then `eval/run_eval.py` and
`eval/run_answer_eval.py` (the latter makes real Gemini calls).

## Ask (answers with citations)
`POST /api/ask {"question": "..."}` retrieves the 8 best transcript chunks, asks Gemini to answer **only** from them and to cite `[n]`, and
returns `{answer, citations: [{n, video_id, title, start_sec, end_sec}], mode: "rag"}`. Citations that do not exist are dropped; when
nothing relevant is found the model is not called at all. Login required, 10 questions per minute per user.

It needs a Gemini API key (not set by default). One-time setup, PowerShell:
```powershell
# 1. add to .env:   GEMINI_API_KEY=your-key        (optional: LLM_MODEL=<model id>, default is the alias gemini-flash-latest)
docker compose up -d api            # recreates the api container so it sees the new environment (no rebuild needed)
docker compose exec api python -m app.scripts.check_llm            # ONE real tiny request: confirms key, model and network
docker compose exec api python -m app.scripts.check_llm --models   # if it says the model was rejected: lists the ids your key can use
```
If a model id is rejected, set `LLM_MODEL` in `.env` to one of the listed ids and run `docker compose up -d api` again. In the app, the
**Ask** tab is a chat: click a numbered chip (or a source under the answer) to play that moment in the player beside it. Without a key,
Ask shows "Ask is not set up yet: an administrator needs to configure GEMINI_API_KEY." instead of failing mysteriously.

## Smoke test the pipeline (PowerShell, stack running)
```powershell
.\scripts\smoke_ingest.ps1 -Email admin@example.com -Password 'ChangeMe123!' -File "sample\purpose.mp4" -ExpectedChunks 59
.\scripts\smoke_ingest.ps1 -Email admin@example.com -Password 'ChangeMe123!' -File "sample\EduSphereDemonstration.mp4" -ExpectNoSpeech
```
It uploads, checks storage by fetching the pre-signed stream URL, polls the job, and prints the chunk count.
A silent video must end as `failed` with "No speech detected in this video." (`-ExpectFail` accepts any failed job, e.g. a renamed .txt).
The pre-signed URL check is a warning, not a failure. Works on Windows PowerShell 5.1.

## Inspect the database
```powershell
docker compose exec postgres psql -U svs -P pager=off -c "select count(*), min(start_sec), max(end_sec) from chunks;"
```
`-P pager=off` stops psql from opening a pager.

Existing databases (created before the `chunks (video_id, idx)` unique constraint) need this once:
```powershell
docker compose exec postgres psql -U svs -P pager=off -c "ALTER TABLE chunks ADD CONSTRAINT chunks_video_id_idx_key UNIQUE (video_id, idx);"
```
If it fails with "could not create unique index", duplicates exist: reprocess the affected videos or delete the extra rows first.

## Endpoints
| Method | Path | Access |
|---|---|---|
| POST | /auth/register, /auth/login | public |
| POST | /auth/refresh, /auth/logout | refresh cookie |
| GET | /auth/me | logged in |
| POST / DELETE | /api/videos (upload takes `file` + optional `title`; mp4/webm/mov/mkv, max `MAX_UPLOAD_MB`) | admin |
| POST | /api/videos/{id}/reprocess (409 while a job is running) | admin |
| GET | /api/videos, /api/videos/{id}/stream, /api/videos/{id}/transcript, /api/jobs/{id}, /api/search | logged in |

## Tests
Integration tests run against a separate `svs_test` database and Redis db 15, so your dev data is never touched.
One-time setup on an existing database volume (fresh volumes create it automatically):
```powershell
docker compose exec postgres psql -U svs -d postgres -P pager=off -c "CREATE DATABASE svs_test;"
docker compose up -d        # recreates api/worker: mounts ./db for the tests, runs Celery beat in the worker
docker compose exec api pytest
```
Without Docker (`cd backend; pytest`), unit tests run and integration tests skip unless Postgres/Redis are reachable on localhost
(`TEST_DATABASE_URL` / `TEST_REDIS_URL` override the defaults).

Auth: passwords need 8+ characters with a letter and a digit. Rate limits (429 + `Retry-After`): login 5/min per IP+email,
search 30/min per user. Expired refresh tokens are deleted hourly.

Existing databases need this once (Phase 3 index):
```powershell
docker compose exec postgres psql -U svs -P pager=off -c "CREATE INDEX IF NOT EXISTS refresh_tokens_user_idx ON refresh_tokens (user_id);"
```

Unit tests only:
```bash
cd backend && pip install -r requirements.txt && pytest
```

## Roadmap
- [x] Wk 1–4: transcription, chunking, pgvector, auth, search API (this scaffold)
- [ ] Wk 5: React frontend (login, admin upload + job polling, search with seeking player)
- [ ] Wk 7–9: `/api/ask` — RAG + hand-rolled Gemini tool-calling agent (`app/agent/`)
- [ ] Wk 10: hybrid search (tsvector + RRF), reranker
- [ ] Wk 11: eval set, Recall@5 / MRR (`backend/eval/`)
- [ ] Wk 12: deploy, demo
