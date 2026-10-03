import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import type { Chunk } from '../api'
import { describeError } from '../api/errors'
import { buildTranscript } from '../lib/transcript'
import type { TranscriptLine } from '../lib/transcript'

interface TranscriptState {
  status: 'loading' | 'ready' | 'error'
  lines: TranscriptLine[]
  error: string | null
}

/** The video's full transcript as non-repeating sentence lines (the API returns overlapping chunks). */
export function useTranscript(videoId: string) {
  const [state, setState] = useState<TranscriptState>({ status: 'loading', lines: [], error: null })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setState({ status: 'loading', lines: [], error: null })
    api
      .request<Chunk[]>(`/api/videos/${encodeURIComponent(videoId)}/transcript`, { signal: controller.signal })
      .then(
        (chunks) => setState({ status: 'ready', lines: buildTranscript(chunks), error: null }),
        (e: unknown) => {
          if (!controller.signal.aborted) setState({ status: 'error', lines: [], error: describeError(e, 'transcript') })
        },
      )
    return () => controller.abort()
  }, [videoId, attempt])

  return { ...state, retry: useCallback(() => setAttempt((a) => a + 1), []) }
}
