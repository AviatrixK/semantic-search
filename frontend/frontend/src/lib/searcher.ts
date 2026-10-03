import type { SearchHit } from '../api/types.js'

export interface SearchState {
  status: 'idle' | 'loading' | 'ready' | 'error'
  /** The query the hits (or the error) belong to. */
  query: string
  hits: SearchHit[]
  error: string | null
  /** Seconds to wait before retrying, when the server rate-limited us. */
  retryAfter: number | null
}

export interface SearcherOptions {
  run: (query: string, signal: AbortSignal) => Promise<SearchHit[]>
  onState: (state: SearchState) => void
  /** Maps a thrown error to text for the screen (and Retry-After seconds for 429). */
  describe: (error: unknown) => { message: string; retryAfter?: number }
  debounceMs?: number
  /** The backend rejects queries shorter than 2 characters. */
  minLength?: number
  /** setTimeout stand-in: returns a cancel function. Injectable for tests. */
  schedule?: (fn: () => void, ms: number) => () => void
}

export interface Searcher {
  /** Call on every keystroke: searches after the user pauses. */
  setQuery(query: string): void
  /** Enter key: searches right away (does nothing if these results are already showing). */
  submit(): void
  dispose(): void
}

const defaultSchedule = (fn: () => void, ms: number) => {
  const id = setTimeout(fn, ms)
  return () => clearTimeout(id)
}

export const IDLE: SearchState = { status: 'idle', query: '', hits: [], error: null, retryAfter: null }

/**
 * Debounced search with cancellation: typing never queues requests (a new one aborts the previous), an unchanged query
 * is not searched twice, and only the newest response may reach the screen. Previous results stay visible while the
 * next ones load.
 */
export function createSearcher(o: SearcherOptions): Searcher {
  const debounceMs = o.debounceMs ?? 400
  const minLength = o.minLength ?? 2
  const schedule = o.schedule ?? defaultSchedule

  let query = ''
  let state: SearchState = IDLE
  let cancelTimer: (() => void) | null = null
  let controller: AbortController | null = null
  let runId = 0
  let disposed = false

  const emit = (next: SearchState) => {
    state = next
    if (!disposed) o.onState(next)
  }
  const stopPending = () => {
    cancelTimer?.()
    cancelTimer = null
    controller?.abort()
    controller = null
    runId++ // invalidates a response that is still on its way
  }

  async function execute() {
    cancelTimer = null
    const q = query
    if (q.length < minLength) return
    if (state.query === q && (state.status === 'ready' || state.status === 'loading')) return // already showing / fetching
    controller?.abort()
    const mine = ++runId
    controller = new AbortController()
    emit({ ...state, status: 'loading', query: q, error: null, retryAfter: null })
    try {
      const hits = await o.run(q, controller.signal)
      if (mine !== runId) return
      emit({ status: 'ready', query: q, hits, error: null, retryAfter: null })
    } catch (error) {
      if (mine !== runId) return // aborted or superseded
      const d = o.describe(error)
      emit({ status: 'error', query: q, hits: state.hits, error: d.message, retryAfter: d.retryAfter ?? null })
    }
  }

  return {
    setQuery(raw) {
      const q = raw.trim().replace(/\s+/g, ' ')
      if (q === query) return
      query = q
      cancelTimer?.()
      cancelTimer = null
      if (q.length < minLength) {
        stopPending()
        emit(IDLE)
        return
      }
      if (state.query === q && state.status === 'ready') {
        stopPending()
        return
      }
      cancelTimer = schedule(() => void execute(), debounceMs)
    },
    submit() {
      cancelTimer?.()
      cancelTimer = null
      if (state.query === query && state.status === 'error') state = { ...state, status: 'idle', query: '' } // retry after an error
      void execute()
    },
    dispose() {
      disposed = true
      stopPending()
    },
  }
}
