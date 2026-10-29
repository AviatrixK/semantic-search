// Shapes returned by the backend. Kept free of runtime code so lib/ modules can import them without pulling in the app client.
export interface User {
  id: string
  email: string
  role: 'user' | 'admin'
}

export interface Video {
  id: string
  title: string
  status: 'uploaded' | 'ready' | 'failed' | string
  duration_sec: number | null
  created_at: string
}

export interface Job {
  id: string
  video_id: string
  stage: 'queued' | 'extracting' | 'transcribing' | 'embedding' | 'done' | 'failed' | string
  progress: number
  error: string | null
}

export interface UploadOut {
  video_id: string
  job_id: string
}

export interface TokenOut {
  access_token: string
  token_type: string
}

export interface SearchHit {
  video_id: string
  title: string
  start_sec: number
  end_sec: number
  text: string
  /** English translation of `text`; null/absent while pending. */
  text_en?: string | null
  /** Latin-script transcription of `text` (not shown yet). */
  text_roman?: string | null
  /** The video's language code ("ta-IN" or "ta"); null when unknown. */
  language?: string | null
  /** Cosine similarity, 0..1 (keyword mode: a 0..1 keyword relevance). */
  score: number
  /** The sentence of `text` that best matches the query (a substring of it), or null. */
  highlight: string | null
  /** Hybrid mode only: which searches returned it ("vector", "keyword"). */
  found_by?: string[] | null
}

/** One transcript chunk (~30 s windows that overlap their neighbours by several seconds). */
export interface Chunk {
  idx: number
  start_sec: number
  end_sec: number
  text: string
}

export interface StreamOut {
  url: string
}
