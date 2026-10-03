/** Upload queue state machine (pure: no React, no network). Files upload one at a time; processing runs in parallel. */
export interface QueueFileLike {
  name: string
  size: number
}

export type Phase = 'queued' | 'uploading' | 'processing' | 'error'

export interface QueueItem<F extends QueueFileLike = QueueFileLike> {
  id: string
  title: string
  fileName: string
  fileSize: number
  /** Dropped after the upload finishes so the browser can free the file. */
  file: F | null
  phase: Phase
  loaded: number
  total: number
  videoId: string | null
  jobId: string | null
  error: string | null
}

export interface QueueState<F extends QueueFileLike = QueueFileLike> {
  items: QueueItem<F>[]
}

export type QueueAction<F extends QueueFileLike = QueueFileLike> =
  | { type: 'enqueue'; entries: { id: string; file: F; title: string }[] }
  | { type: 'start'; id: string }
  | { type: 'progress'; id: string; loaded: number; total: number }
  | { type: 'uploaded'; id: string; videoId: string; jobId: string }
  | { type: 'failed'; id: string; message: string }
  | { type: 'remove'; id: string }
  | { type: 'removeByVideo'; videoId: string }
  /** A job started elsewhere (e.g. reprocess) that should get a status card too. */
  | { type: 'track'; id: string; videoId: string; jobId: string; title: string }

export function initialQueue<F extends QueueFileLike = QueueFileLike>(): QueueState<F> {
  return { items: [] }
}

function update<F extends QueueFileLike>(
  state: QueueState<F>, id: string, allowed: Phase[], patch: (item: QueueItem<F>) => Partial<QueueItem<F>>,
): QueueState<F> {
  let changed = false
  const items = state.items.map((item) => {
    if (item.id !== id || !allowed.includes(item.phase)) return item
    changed = true
    return { ...item, ...patch(item) }
  })
  return changed ? { items } : state // unchanged state keeps React from re-rendering
}

export function queueReducer<F extends QueueFileLike>(state: QueueState<F>, action: QueueAction<F>): QueueState<F> {
  switch (action.type) {
    case 'enqueue':
      return {
        items: [
          ...state.items,
          ...action.entries.map<QueueItem<F>>((e) => ({
            id: e.id, title: e.title, fileName: e.file.name, fileSize: e.file.size, file: e.file,
            phase: 'queued', loaded: 0, total: e.file.size, videoId: null, jobId: null, error: null,
          })),
        ],
      }
    case 'start': // idempotent: only a queued item can start (StrictMode may fire the effect twice)
      return update(state, action.id, ['queued'], () => ({ phase: 'uploading', loaded: 0 }))
    case 'progress':
      return update(state, action.id, ['uploading'], () => ({ loaded: action.loaded, total: action.total }))
    case 'uploaded':
      return update(state, action.id, ['uploading'], (i) => ({
        phase: 'processing', file: null, loaded: i.total, videoId: action.videoId, jobId: action.jobId,
      }))
    case 'failed':
      return update(state, action.id, ['queued', 'uploading'], () => ({ phase: 'error', file: null, error: action.message }))
    case 'remove':
      return { items: state.items.filter((i) => i.id !== action.id) }
    case 'removeByVideo':
      return { items: state.items.filter((i) => i.videoId !== action.videoId) }
    case 'track':
      return {
        items: [
          ...state.items,
          {
            id: action.id, title: action.title, fileName: action.title, fileSize: 0, file: null, phase: 'processing',
            loaded: 0, total: 0, videoId: action.videoId, jobId: action.jobId, error: null,
          },
        ],
      }
  }
}

/** Id of the next file to upload: the first queued one, but only while nothing else is uploading. */
export function nextToStart<F extends QueueFileLike>(state: QueueState<F>): string | undefined {
  if (state.items.some((i) => i.phase === 'uploading')) return undefined
  return state.items.find((i) => i.phase === 'queued')?.id
}

/** True while files are waiting or being sent (closing the tab would lose them). */
export function hasPendingUploads<F extends QueueFileLike>(state: QueueState<F>): boolean {
  return state.items.some((i) => i.phase === 'queued' || i.phase === 'uploading')
}
