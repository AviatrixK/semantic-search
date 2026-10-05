import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { SearchHit } from '../api'
import { ApiError, describeError } from '../api/errors'
import { IDLE, createSearcher } from '../lib/searcher'
import type { SearchState, Searcher } from '../lib/searcher'
import { searchUrl } from '../lib/searchParams'
import type { SearchFilters } from '../lib/searchParams'

const RESULTS_PER_SEARCH = 10

/**
 * Debounced (400 ms), cancellable search over GET /api/search. A change of mode or filter searches again at once.
 * See lib/searcher.ts for the rules. `initial` is only read on first render (the page keeps its own copy for the controls).
 */
export function useSearch(initial: SearchFilters) {
  const [state, setState] = useState<SearchState>(IDLE)
  const searcher = useRef<Searcher | null>(null)
  const start = useRef(initial)

  useEffect(() => {
    const s = createSearcher({
      filters: start.current,
      run: (q, signal, filters) => api.request<SearchHit[]>(searchUrl(q, RESULTS_PER_SEARCH, filters), { signal }),
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
    setFilters: useCallback((f: SearchFilters) => searcher.current?.setFilters(f), []),
    submit: useCallback(() => searcher.current?.submit(), []),
  }
}
