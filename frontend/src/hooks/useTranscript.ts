import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import type { Chunk, Transcript } from '../api'
import { describeError } from '../api/errors'

interface TranscriptState {
  status: 'loading' | 'ready' | 'error'
  chunks: Chunk[]
  /** The video's language code, null when unknown. */
  language: string | null
  error: string | null
}

const LOADING: TranscriptState = { status: 'loading', chunks: [], language: null, error: null }

/**
 * The video's transcript as the API returns it: overlapping chunks with their original, English and romanised text, plus the video's
 * language. `buildTranscript` turns them into non-repeating lines in whichever language is shown.
 */
export function useTranscript(videoId: string) {
  const [state, setState] = useState<TranscriptState>(LOADING)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setState(LOADING)
    api
      .request<Transcript>(`/api/videos/${encodeURIComponent(videoId)}/transcript`, { signal: controller.signal })
      .then(
        (t) => setState({ status: 'ready', chunks: t.chunks, language: t.language, error: null }),
        (e: unknown) => {
          if (!controller.signal.aborted) setState({ status: 'error', chunks: [], language: null, error: describeError(e, 'transcript') })
        },
      )
    return () => controller.abort()
  }, [videoId, attempt])

  return { ...state, retry: useCallback(() => setAttempt((a) => a + 1), []) }
}
