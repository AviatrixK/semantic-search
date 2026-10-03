import { createStreamUrlCache } from '../lib/streamUrls'
import { api } from './index'
import type { StreamOut } from './types'

/** Presigned video URLs, cached per video (see lib/streamUrls.ts). */
export const streamUrls = createStreamUrlCache(async (videoId) => {
  const { url } = await api.request<StreamOut>(`/api/videos/${encodeURIComponent(videoId)}/stream`)
  return url
})
