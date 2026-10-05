import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { Video } from '../api'
import ResultCard from '../components/ResultCard'
import SearchOptions from '../components/SearchOptions'
import { ResultSkeletons } from '../components/Skeleton'
import { useCountdown } from '../hooks/useCountdown'
import { useSearch } from '../hooks/useSearch'
import { useVideos } from '../hooks/useVideos'
import { MODE_OPTIONS, filtersKey, filtersToParams, modeOfKey, parseFilters } from '../lib/searchParams'
import type { SearchFilters } from '../lib/searchParams'
import styles from './Search.module.css'

/**
 * /search?q=...&mode=...&video=...&after=...: YouTube-style results. The query comes from the search box in the header; mode and
 * filters sit above the list as chips. The address is the single source of truth, so reload, back and shared links all work.
 */
export default function Search() {
  const [params, setParams] = useSearchParams()
  const q = (params.get('q') ?? '').trim()
  const filters = useMemo(() => parseFilters(params), [params])
  const fkey = filtersKey(filters)
  const { state, setQuery, setFilters, submit } = useSearch(filters)
  const { videos } = useVideos()
  const ready = useMemo(() => videos?.filter((v) => v.status === 'ready') ?? null, [videos])
  const byId = useMemo(() => new Map<string, Video>((videos ?? []).map((v) => [v.id, v])), [videos])

  // New query in the address: search at once (no debounce: the header box only sends it on Enter).
  useEffect(() => {
    setQuery(q)
    submit()
  }, [q, setQuery, submit])

  // A mode or filter chip changed: the hook searches the current query again straight away.
  useEffect(() => {
    setFilters(filters)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the key stands for the whole filters object
  }, [fkey, setFilters])

  // Rate limited: count down, then retry once automatically.
  const [retryAt, setRetryAt] = useState<number | null>(null)
  const secondsLeft = useCountdown(retryAt)
  useEffect(() => {
    setRetryAt(state.status === 'error' && state.retryAfter ? Date.now() + state.retryAfter * 1000 : null)
  }, [state.status, state.retryAfter])
  useEffect(() => {
    if (retryAt !== null && secondsLeft === 0) {
      setRetryAt(null)
      submit()
    }
  }, [retryAt, secondsLeft, submit])

  function changeFilters(next: SearchFilters) {
    setParams({ ...(q ? { q } : {}), ...filtersToParams(next) }, { replace: true })
  }

  const hasHits = state.hits.length > 0
  const loading = state.status === 'loading'
  const shownMode = state.filtersKey ? modeOfKey(state.filtersKey) : filters.mode // what the results on screen were searched with

  return (
    <main className={styles.page}>
      <h1 className="sr-only">Search results{q ? ` for ${q}` : ''}</h1>
      <SearchOptions filters={filters} onChange={changeFilters} videos={ready} />

      <div className={styles.status} aria-live="polite">
        {state.status === 'ready' && hasHits && (
          <span>
            {state.hits.length} moment{state.hits.length === 1 ? '' : 's'} for “{state.query}” · {MODE_OPTIONS.find((m) => m.value === shownMode)?.label}
          </span>
        )}
      </div>

      {q.length < 2 && (
        <p className={styles.hint}>Type at least 2 characters in the search box above and press Enter. Describe an idea in your own words, or type a name exactly as it is said.</p>
      )}

      {loading && !hasHits && q.length >= 2 && <ResultSkeletons count={4} />}

      {state.status === 'ready' && !hasHits && (
        <div className={styles.empty}>
          <strong>No matches for “{state.query}”.</strong>
          <span>
            {shownMode === 'keyword'
              ? 'Exact-word search needs every word to appear in the same moment. Try fewer words, or switch to Best match.'
              : 'Try different words or a more general description. Weak matches are hidden on purpose.'}
          </span>
        </div>
      )}

      {state.status === 'error' && (
        <div className={styles.error} role="alert">
          <span>{state.error}</span>
          {retryAt !== null && secondsLeft > 0 ? (
            <span>Trying again automatically in {secondsLeft}s…</span>
          ) : (
            <button type="button" className="btn btn-secondary" onClick={submit}>Try again</button>
          )}
        </div>
      )}

      {hasHits && (
        <ul className={`${styles.list} ${loading ? styles.stale : ''}`} aria-label="Search results" aria-busy={loading}>
          {state.hits.map((hit) => (
            <li key={`${hit.video_id}:${hit.start_sec}`}>
              <ResultCard hit={hit} video={byId.get(hit.video_id)} mode={shownMode} />
            </li>
          ))}
        </ul>
      )}
    </main>
  )
}
