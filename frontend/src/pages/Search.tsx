import { useCallback, useEffect, useId, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import type { SearchHit } from '../api'
import ResultCard from '../components/ResultCard'
import Spinner from '../components/Spinner'
import VideoPlayer from '../components/VideoPlayer'
import type { SeekRequest } from '../components/VideoPlayer'
import { useSearch } from '../hooks/useSearch'
import { useCountdown } from '../hooks/useCountdown'
import { formatDuration } from '../lib/format'
import { moveActive } from '../lib/listNav'
import { watchPath } from '../lib/timeParam'
import styles from './Search.module.css'

const hitKey = (h: SearchHit) => `${h.video_id}:${h.start_sec}`

export default function Search() {
  const [params, setParams] = useSearchParams()
  const initialQuery = params.get('q') ?? ''
  const [input, setInput] = useState(initialQuery)
  const { state, setQuery, submit } = useSearch()
  const [active, setActive] = useState(-1) // keyboard selection
  const [selected, setSelected] = useState<SearchHit | null>(null) // what the player shows
  const [request, setRequest] = useState<SeekRequest | null>(null)
  const nonce = useRef(0)
  const playerBox = useRef<HTMLElement>(null)
  const listId = useId()
  const optionId = (i: number) => `${listId}-option-${i}`

  // A shared link /?q=... searches straight away (no debounce).
  useEffect(() => {
    if (initialQuery.trim().length >= 2) {
      setQuery(initialQuery)
      submit()
    }
  }, [])

  // Keep the address bar in step with the query being shown, so reload / back / sharing keep it.
  useEffect(() => {
    if (state.status !== 'idle' && state.query) setParams({ q: state.query }, { replace: true })
  }, [state.status, state.query, setParams])

  // New results start with nothing selected.
  useEffect(() => setActive(-1), [state.hits])

  // Keep the arrow-key selection visible.
  useEffect(() => {
    if (active >= 0) document.getElementById(optionId(active))?.scrollIntoView({ block: 'nearest' })
  }, [active])

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

  const play = useCallback((hit: SearchHit, index: number) => {
    setActive(index)
    setSelected(hit)
    setRequest({ videoId: hit.video_id, t: hit.start_sec, play: true, nonce: ++nonce.current })
    playerBox.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }) // matters when the layout is stacked (mobile)
  }, [])

  function onChange(value: string) {
    setInput(value)
    setQuery(value)
    if (value.trim().length < 2 && params.get('q')) setParams({}, { replace: true })
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.nativeEvent.isComposing) return // an IME is still composing text
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (state.hits.length === 0) return
      e.preventDefault()
      setActive((cur) => moveActive(cur, e.key as 'ArrowDown' | 'ArrowUp', state.hits.length))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      const hit = state.hits[active]
      if (hit) play(hit, active)
      else submit()
    } else if (e.key === 'Escape') {
      setActive(-1)
    }
  }

  const trimmed = input.trim()
  const hasHits = state.hits.length > 0
  const loading = state.status === 'loading'

  return (
    <main className={styles.page}>
      <div className={styles.layout}>
        <section className={styles.results} aria-label="Search">
          <form
            role="search"
            className={styles.form}
            onSubmit={(e) => {
              e.preventDefault()
              submit()
            }}
          >
            <input
              className={`input ${styles.box}`}
              type="search"
              name="q"
              value={input}
              onChange={(e) => onChange(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Search by meaning, e.g. how to sound more confident"
              aria-label="Search videos"
              role="combobox"
              aria-expanded={hasHits}
              aria-controls={listId}
              aria-autocomplete="none"
              aria-activedescendant={active >= 0 ? optionId(active) : undefined}
              autoComplete="off"
              autoFocus
            />
            <button type="submit" className="btn btn-primary" disabled={trimmed.length < 2 || loading}>
              {loading ? 'Searching…' : 'Search'}
            </button>
          </form>

          <div className={styles.status} aria-live="polite">
            {loading && !hasHits && <Spinner label="Searching…" />}
            {state.status === 'ready' && hasHits && (
              <span className={styles.count}>
                {state.hits.length} result{state.hits.length === 1 ? '' : 's'} · ↑ ↓ to move, Enter to play
              </span>
            )}
          </div>

          {state.status === 'idle' && trimmed.length === 0 && (
            <p className={styles.hint}>
              Describe what you are looking for in your own words. Results are matched by meaning, not by exact words, and
              clicking one plays the video at that moment.
            </p>
          )}
          {state.status === 'idle' && trimmed.length === 1 && <p className={styles.hint}>Keep typing: at least 2 characters.</p>}

          {state.status === 'ready' && !hasHits && (
            <div className={styles.empty}>
              <strong>No matches for “{state.query}”.</strong>
              <span>Try different words or a more general description. Weak matches are hidden on purpose.</span>
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
            <ul id={listId} role="listbox" aria-label="Search results" aria-busy={loading} className={`${styles.list} ${loading ? styles.stale : ''}`}>
              {state.hits.map((hit, i) => (
                <ResultCard
                  key={hitKey(hit)}
                  id={optionId(i)}
                  hit={hit}
                  active={i === active}
                  playing={selected !== null && hitKey(selected) === hitKey(hit)}
                  onPlay={() => play(hit, i)}
                />
              ))}
            </ul>
          )}
        </section>

        <aside className={styles.player} ref={playerBox} data-active={request ? '' : undefined} aria-label="Video player">
          <VideoPlayer request={request} label={selected?.title ?? 'Video player'} emptyMessage="Pick a result to play it here." />
          {selected && (
            <div className={styles.now}>
              <strong title={selected.title}>{selected.title}</strong>
              <span>{formatDuration(selected.start_sec)} – {formatDuration(selected.end_sec)}</span>
              <Link to={watchPath(selected.video_id, selected.start_sec)}>Open with full transcript</Link>
            </div>
          )}
        </aside>
      </div>
    </main>
  )
}
