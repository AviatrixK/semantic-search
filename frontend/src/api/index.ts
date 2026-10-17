import { createApiClient, webLocksRunner } from './client'

/** The app-wide client. The access token lives only in this module's memory (never localStorage). */
export const api = createApiClient({
  baseUrl: import.meta.env.VITE_API_BASE ?? '',
  runExclusive: webLocksRunner(),
})

export type { Chunk, Job, SearchHit, StreamOut, TokenOut, UploadOut, User, Video } from './types'
