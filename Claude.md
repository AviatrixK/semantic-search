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
  returns 0 segments. "No speech detected" must be a clear message in the CLI (Phase 1) and a failed job with a readable
  error in the worker (Phase 2), never an empty transcript or a "done" job with zero chunks. `sample/` is gitignored.
- **Dev machine:** Windows + PowerShell, Python 3.13 in `backend\.venv`, Docker Desktop (WSL 2), ffmpeg on PATH. The project
  path contains spaces: quote every path in commands and scripts. Internet is slow/unreliable: avoid forcing large
  re-downloads and ask before adding a heavy dependency.

## Commands
- Stack: `docker compose up --build`
- Admin: `docker compose exec api python -m app.scripts.create_admin <email> <password>`
- Tests: `cd backend; pytest`
- Frontend: `cd frontend; npm run dev` (http://localhost:5173)