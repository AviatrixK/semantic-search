import { useCallback, useEffect, useReducer, useRef } from 'react'
import { api } from '../api'
import type { UploadOut } from '../api'
import { describeError } from '../api/errors'
import { hasPendingUploads, initialQueue, nextToStart, queueReducer } from '../lib/uploadQueue'
import type { QueueAction, QueueState } from '../lib/uploadQueue'
import { buildUploadForm } from '../lib/uploadValidation'

const PROGRESS_THROTTLE_MS = 100
let counter = 0
const newId = () => `u${Date.now().toString(36)}-${++counter}`
const reducer = queueReducer as (s: QueueState<File>, a: QueueAction<File>) => QueueState<File>

/** Several files can be queued: they upload one at a time (sharing bandwidth would slow every one), while the
 *  server-side processing of earlier files runs in parallel and is shown by their own job cards. */
export function useUploadQueue(onUploaded?: () => void) {
  const [state, dispatch] = useReducer(reducer, undefined, () => initialQueue<File>())
  const started = useRef(new Set<string>()) // guards against StrictMode running the effect twice
  const controllers = useRef(new Map<string, AbortController>())
  const onUploadedRef = useRef(onUploaded)
  onUploadedRef.current = onUploaded

  useEffect(() => {
    const id = nextToStart(state)
    if (!id || started.current.has(id)) return
    const item = state.items.find((i) => i.id === id)
    if (!item?.file) return
    started.current.add(id)
    dispatch({ type: 'start', id })

    const controller = new AbortController()
    controllers.current.set(id, controller)
    let last = 0
    api
      .upload<UploadOut>('/api/videos', buildUploadForm(item.file, item.title), {
        signal: controller.signal,
        onProgress: (loaded, total) => {
          const now = Date.now()
          if (loaded < total && now - last < PROGRESS_THROTTLE_MS) return
          last = now
          dispatch({ type: 'progress', id, loaded, total })
        },
      })
      .then((res) => {
        dispatch({ type: 'uploaded', id, videoId: res.video_id, jobId: res.job_id })
        onUploadedRef.current?.()
      })
      .catch((e: unknown) => {
        if (controller.signal.aborted) return // cancelled by the user: the card was already removed
        dispatch({ type: 'failed', id, message: describeError(e, 'upload') })
      })
      .finally(() => controllers.current.delete(id))
  }, [state])

  // Leaving the page would abort files that are still waiting or uploading.
  const pending = hasPendingUploads(state)
  useEffect(() => {
    if (!pending) return
    const warn = (e: BeforeUnloadEvent) => e.preventDefault()
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [pending])

  return {
    items: state.items,
    add: useCallback((entries: { file: File; title: string }[]) => {
      dispatch({ type: 'enqueue', entries: entries.map((e) => ({ ...e, id: newId() })) })
    }, []),
    /** Removes a card; if it is uploading right now, the upload is aborted. */
    cancel: useCallback((id: string) => {
      controllers.current.get(id)?.abort()
      dispatch({ type: 'remove', id })
    }, []),
    dismiss: useCallback((id: string) => dispatch({ type: 'remove', id }), []),
    dismissByVideo: useCallback((videoId: string) => dispatch({ type: 'removeByVideo', videoId }), []),
    /** Show a status card for a job started elsewhere (reprocess). */
    track: useCallback((videoId: string, jobId: string, title: string) => {
      dispatch({ type: 'track', id: newId(), videoId, jobId, title })
    }, []),
  }
}
