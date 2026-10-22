-- text_en / text_roman made by a Gemini Flash-Lite model (TEXT_VARIANT_PROVIDER=gemini): per-video status and token counts.
-- Idempotent: safe to run twice. db/init.sql already contains this for fresh databases.
--
--   docker compose exec -T postgres psql -U svs -d svs -P pager=off -v ON_ERROR_STOP=1 < db/migrations/2026-10-22-text-variants.sql
BEGIN;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS text_variants_status TEXT;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS text_variants_provider TEXT;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS text_variants_model TEXT;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS text_variants_in_tokens INT;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS text_variants_out_tokens INT;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS text_variants_est_tokens INT;
COMMIT;
