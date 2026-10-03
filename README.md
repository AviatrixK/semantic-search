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

## Smoke test the pipeline (PowerShell, stack running)
```powershell
.\scripts\smoke_ingest.ps1 -Email admin@example.com -Password 'ChangeMe123!' -File "sample\purpose.mp4" -ExpectedChunks 59
.\scripts\smoke_ingest.ps1 -Email admin@example.com -Password 'ChangeMe123!' -File "sample\EduSphereDemonstration.mp4" -ExpectNoSpeech
```
It uploads, checks storage by fetching the pre-signed stream URL, polls the job, and prints the chunk count.
A silent video must end as `failed` with "No speech detected".

## Endpoints
| Method | Path | Access |
|---|---|---|
| POST | /auth/register, /auth/login | public |
| POST | /auth/refresh, /auth/logout | refresh cookie |
| GET | /auth/me | logged in |
| POST / DELETE | /api/videos (upload takes `file` + optional `title`; mp4/webm/mov/mkv, max `MAX_UPLOAD_MB`) | admin |
| POST | /api/videos/{id}/reprocess | admin |
| GET | /api/videos, /api/videos/{id}/stream, /api/videos/{id}/transcript, /api/jobs/{id}, /api/search | logged in |

## Tests
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
