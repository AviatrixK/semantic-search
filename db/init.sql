-- Runs automatically the first time the Postgres volume is created.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
  id             UUID PRIMARY KEY,
  email          TEXT UNIQUE NOT NULL,
  password_hash  TEXT NOT NULL,
  role           TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE refresh_tokens (
  id          UUID PRIMARY KEY,
  user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash  TEXT NOT NULL UNIQUE,
  expires_at  TIMESTAMPTZ NOT NULL,
  revoked     BOOLEAN NOT NULL DEFAULT false
);

CREATE INDEX refresh_tokens_user_idx ON refresh_tokens (user_id);

CREATE TABLE videos (
  id            UUID PRIMARY KEY,
  title         TEXT NOT NULL,
  storage_key   TEXT NOT NULL,
  duration_sec  INT,
  language      TEXT,
  status        TEXT NOT NULL DEFAULT 'uploaded',
  uploaded_by   UUID REFERENCES users(id),
  sha256        TEXT,   -- SHA-256 of the uploaded file; unique where set (NULL for videos that predate the column)
  language_hint TEXT,   -- language the admin chose on upload (Sarvam code such as hi-IN); NULL = detect automatically
  transcriber   TEXT,   -- engine that produced the current chunks: whisper | sarvam (NULL = whisper, older videos)
  sarvam_job_id     TEXT,  -- the Sarvam batch speech-to-text job for this video: a retry resumes it instead of starting (and paying for) a new one
  sarvam_job_status TEXT,  -- created | uploaded | started | the job states Sarvam reports (accepted, pending, running, completed, failed) | consumed
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX videos_sha256_key ON videos (sha256);

CREATE TABLE jobs (
  id          UUID PRIMARY KEY,
  video_id    UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  stage       TEXT NOT NULL DEFAULT 'queued',  -- queued|extracting|transcribing|embedding|done|failed
  progress    INT NOT NULL DEFAULT 0,
  error       TEXT,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id          UUID PRIMARY KEY,
  video_id    UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  idx         INT NOT NULL,
  start_sec   REAL NOT NULL,
  end_sec     REAL NOT NULL,
  text        TEXT NOT NULL,      -- what is shown: the spoken text in its own language/script
  text_native TEXT,               -- same as text (own script); kept separate so the three variants are explicit
  text_roman  TEXT,               -- romanised (Latin script) transcription, e.g. Hindi written as 'mera phone number hai'
  text_en     TEXT,               -- English translation
  embedding   VECTOR(384),        -- LOCAL model only (EMBED_PROVIDER=local); must match EMBED_DIM. NULL for hosted providers
  -- Two keyword indexes (see Claude.md): tsv = 'simple' (no stemming, any script) over the native text and its romanisation;
  -- tsv_en = 'english' (stemming) over the English text. Older Whisper videos have no text_en: their text is the English one.
  tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text_native, text) || ' ' || coalesce(text_roman, ''))) STORED,
  tsv_en      TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(text_en, text))) STORED,
  CONSTRAINT chunks_video_id_idx_key UNIQUE (video_id, idx)
);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX chunks_tsv_en_gin ON chunks USING gin (tsv_en);
CREATE INDEX chunks_video_idx ON chunks (video_id);

-- Vectors of the HOSTED embedding providers (and any model compared in the eval). One row per chunk, model and text variant.
-- `embedding` has no fixed size on purpose: every model has its own. pgvector's HNSW index needs a fixed size, so each model gets
-- its own partial index on embedding::vector(<dim>), created by app/services/vectorstore.py::ensure_index.
CREATE TABLE chunk_embeddings (
  chunk_id   UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  model      TEXT NOT NULL,       -- '<provider>/<model>@<dimensions>', e.g. gemini/gemini-embedding-2@1536
  field      TEXT NOT NULL CHECK (field IN ('native', 'en')),
  embedding  VECTOR NOT NULL,
  PRIMARY KEY (chunk_id, model, field)
);
CREATE INDEX chunk_embeddings_model_idx ON chunk_embeddings (model);

-- One row per sentence of multi-sentence chunks; powers search highlights without embedding at query time.
CREATE TABLE chunk_sentences (
  id          UUID PRIMARY KEY,
  chunk_id    UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  idx         INT NOT NULL,
  text        TEXT NOT NULL,
  embedding   VECTOR(384) NOT NULL,   -- must match EMBED_DIM
  CONSTRAINT chunk_sentences_chunk_id_idx_key UNIQUE (chunk_id, idx)
);

CREATE TABLE search_logs (
  id          UUID PRIMARY KEY,
  user_id     UUID REFERENCES users(id) ON DELETE SET NULL,
  query       TEXT NOT NULL,
  mode        TEXT NOT NULL,   -- search (old rows may say 'ask': written by the removed Ask feature, kept as history)
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
