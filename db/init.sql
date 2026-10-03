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
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
  text        TEXT NOT NULL,
  embedding   VECTOR(384) NOT NULL,   -- must match EMBED_DIM
  tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  CONSTRAINT chunks_video_id_idx_key UNIQUE (video_id, idx)
);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX chunks_video_idx ON chunks (video_id);

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
  mode        TEXT NOT NULL,   -- search|ask
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
