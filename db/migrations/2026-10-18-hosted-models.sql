-- Hosted transcription (Sarvam) and hosted embeddings (Gemini / Cohere / OpenAI) next to the local models.
-- Idempotent: safe to run twice. db/init.sql already contains all of this for fresh databases. Nothing is dropped except the old
-- `tsv` column, which is rebuilt (and only when it is still the old 'english' one).
--
--   docker compose --env-file .env.prod -f docker-compose.prod.yml exec -T postgres \
--     psql -U svs -d svs -P pager=off -v ON_ERROR_STOP=1 < db/migrations/2026-10-18-hosted-models.sql
--
-- The tsv rebuild rewrites the chunks table once and holds a lock while it does (seconds for a small library).
BEGIN;

ALTER TABLE videos ADD COLUMN IF NOT EXISTS language_hint TEXT;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS transcriber TEXT;

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_native TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_roman TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_en TEXT;
UPDATE chunks SET text_native = text WHERE text_native IS NULL;
ALTER TABLE chunks ALTER COLUMN embedding DROP NOT NULL;  -- hosted providers keep their vectors in chunk_embeddings

-- Keyword search: the 'simple' configuration (no stemming, no stop words, works for every script) over all text variants.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_attrdef d
                 JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
                 WHERE d.adrelid = 'chunks'::regclass AND a.attname = 'tsv' AND pg_get_expr(d.adbin, d.adrelid) LIKE '%simple%') THEN
    DROP INDEX IF EXISTS chunks_tsv_gin;
    ALTER TABLE chunks DROP COLUMN IF EXISTS tsv;
    ALTER TABLE chunks ADD COLUMN tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text_native, text) || ' ' || coalesce(text_roman, '') || ' ' || coalesce(text_en, ''))) STORED;
    CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id   UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  model      TEXT NOT NULL,
  field      TEXT NOT NULL CHECK (field IN ('native', 'en')),
  embedding  VECTOR NOT NULL,
  PRIMARY KEY (chunk_id, model, field)
);
CREATE INDEX IF NOT EXISTS chunk_embeddings_model_idx ON chunk_embeddings (model);

COMMIT;
