-- Sarvam refinement: one speech-to-text job per video (job id + status kept on the video so a retry resumes it) and two keyword
-- indexes: tsv ('simple' over native + romanised text) and tsv_en ('english', stemmed, over the English text).
-- Idempotent: safe to run twice. db/init.sql already contains all of this for fresh databases.
--
--   docker compose exec -T postgres psql -U svs -d svs -P pager=off -v ON_ERROR_STOP=1 < db/migrations/2026-10-20-sarvam-single-job.sql
--   (production: docker compose --env-file .env.prod -f docker-compose.prod.yml exec -T postgres psql -U svs -d svs ...)
--
-- Rebuilding the generated columns rewrites the chunks table once and holds a lock while it does (seconds for a small library).
BEGIN;

ALTER TABLE videos ADD COLUMN IF NOT EXISTS sarvam_job_id TEXT;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS sarvam_job_status TEXT;

-- tsv: only native + romanised now (it used to include text_en too).
DO $$
DECLARE expr TEXT;
BEGIN
  SELECT pg_get_expr(d.adbin, d.adrelid) INTO expr FROM pg_attrdef d JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
   WHERE d.adrelid = 'chunks'::regclass AND a.attname = 'tsv';
  IF expr IS NULL OR expr LIKE '%text_en%' OR expr NOT LIKE '%text_roman%' THEN
    DROP INDEX IF EXISTS chunks_tsv_gin;
    ALTER TABLE chunks DROP COLUMN IF EXISTS tsv;
    ALTER TABLE chunks ADD COLUMN tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text_native, text) || ' ' || coalesce(text_roman, ''))) STORED;
  END IF;
END $$;
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv_en TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(text_en, text))) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_en_gin ON chunks USING gin (tsv_en);

COMMIT;
