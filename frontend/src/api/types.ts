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
  /** Cosine similarity, 0..1. */
  score: number
  /** The sentence of `text` that best matches the query (a substring of it), or null. */
  highlight: string | null
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

/** A source of an /api/ask answer: the video moment behind a [n] marker. */
export interface Citation {
  n: number
  video_id: string
  title: string
  start_sec: number
  end_sec: number
}

export interface AskOut {
  answer: string
  citations: Citation[]
  mode: 'rag'
}
