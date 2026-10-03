import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { SearchHit } from '../api'
import { ApiError, describeError } from '../api/errors'
import { IDLE, createSearcher } from '../lib/searcher'
import type { SearchState, Searcher } from '../lib/searcher'

const RESULTS_PER_SEARCH = 10

/** Debounced (400 ms), cancellable search over GET /api/search. See lib/searcher.ts for the rules. */
export function useSearch() {
  const [state, setState] = useState<SearchState>(IDLE)
  const searcher = useRef<Searcher | null>(null)

  useEffect(() => {
    const s = createSearcher({
      run: (q, signal) =>
        api.request<SearchHit[]>(`/api/search?${new URLSearchParams({ q, k: String(RESULTS_PER_SEARCH) })}`, { signal }),
      onState: setState,
      describe: (e) => ({ message: describeError(e, 'search'), retryAfter: e instanceof ApiError ? e.retryAfter : undefined }),
    })
    searcher.current = s
    return () => {
      s.dispose()
      searcher.current = null
    }
  }, [])

  return {
    state,
    setQuery: useCallback((q: string) => searcher.current?.setQuery(q), []),
    submit: useCallback(() => searcher.current?.submit(), []),
  }
}
