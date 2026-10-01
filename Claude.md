# CLAUDE.md — Semantic Video Search

## What this is
Agentic RAG over video. Admin uploads video → Celery worker runs ffmpeg → faster-whisper → chunking
(~30s windows, 5s overlap) → sentence-transformers embeddings → Postgres+pgvector. Users search by meaning
and jump to timestamps; a Gemini tool-calling agent answers multi-step questions with citations.

## Stack (do not change without asking)
- Backend: Python 3.11, FastAPI, SQLAlchemy 2.0, Pydantic v2, pydantic-settings
- DB: Postgres 16 + pgvector (schema in `db/init.sql`; HNSW cosine index; generated `tsv` column)
- Queue: Celery + Redis. Storage: MinIO (S3 API via boto3)
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
- Work one phase at a time from `docs/BUILD_PHASES.md`. Do not start the next phase unprompted.
- Write or update tests for every service you touch. Run `pytest` before saying you're done.
- Keep diffs focused. Don't refactor unrelated code.
- At the end of each phase: summarize files changed, how to test, and propose the commit message.
- Developer is on Windows (PowerShell) with Python 3.13 locally; Docker uses 3.11. Give PowerShell commands.

## Commands
- Stack: `docker compose up --build`
- Admin: `docker compose exec api python -m app.scripts.create_admin <email> <password>`
- Tests: `cd backend; pytest`
- Frontend: `cd frontend; npm run dev` (http://localhost:5173)