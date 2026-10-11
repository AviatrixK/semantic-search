-- Duplicate upload detection: videos.sha256 = SHA-256 (hex) of the uploaded file. Nullable (old videos have none until
-- backfilled), unique where set. Idempotent: safe to run twice. db/init.sql already contains this for fresh databases.
--
--   docker compose --env-file .env.prod -f docker-compose.prod.yml exec -T postgres \
--     psql -U svs -d svs -P pager=off -v ON_ERROR_STOP=1 < db/migrations/2026-10-12-videos-sha256.sql
--
-- Then, BEFORE remuxing old videos to faststart (that rewrites the stored bytes), fill the hashes of existing videos:
--   ... exec api python -m app.scripts.backfill_sha256 --dry-run
--   ... exec api python -m app.scripts.backfill_sha256
BEGIN;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS sha256 TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS videos_sha256_key ON videos (sha256);
COMMIT;
