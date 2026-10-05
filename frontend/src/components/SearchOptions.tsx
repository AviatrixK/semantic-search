import { useId } from 'react'
import type { Video } from '../api'
import { MODE_OPTIONS, activeFilterCount, isValidDate } from '../lib/searchParams'
import type { SearchFilters } from '../lib/searchParams'
import styles from './SearchOptions.module.css'

interface Props {
  filters: SearchFilters
  onChange: (next: SearchFilters) => void
  /** Ready videos for the "Video" filter; null while loading or if they could not be loaded (the filter is then hidden). */
  videos: Video[] | null
}

/** Search mode (best match / meaning / exact words) and the optional video and upload-date filters. */
export default function SearchOptions({ filters, onChange, videos }: Props) {
  const id = useId()
  const hintId = `${id}-hint`
  const hint = MODE_OPTIONS.find((m) => m.value === filters.mode)?.hint
  const filterCount = activeFilterCount(filters)
  const known = videos?.some((v) => v.id === filters.videoId) ?? false

  return (
    <div className={styles.wrap}>
      <fieldset className={styles.modes} aria-describedby={hintId}>
        <legend className="sr-only">Search mode</legend>
        {MODE_OPTIONS.map((m) => (
          <label key={m.value} className={styles.mode}>
            <input
              type="radio"
              name={`${id}-mode`}
              value={m.value}
              checked={filters.mode === m.value}
              onChange={() => onChange({ ...filters, mode: m.value })}
            />
            <span>{m.label}</span>
          </label>
        ))}
      </fieldset>

      <div className={styles.filters}>
        {videos && videos.length > 0 && (
          <label className={styles.field}>
            <span>Video</span>
            <select
              className={`input ${styles.select}`}
              value={filters.videoId ?? ''}
              onChange={(e) => onChange({ ...filters, videoId: e.target.value || null })}
            >
              <option value="">All videos</option>
              {filters.videoId && !known && <option value={filters.videoId}>Selected video</option>}
              {videos.map((v) => (
                <option key={v.id} value={v.id}>{v.title}</option>
              ))}
            </select>
          </label>
        )}
        <label className={styles.field}>
          <span>Uploaded after</span>
          <input
            type="date"
            className={`input ${styles.date}`}
            value={filters.after ?? ''}
            onChange={(e) => onChange({ ...filters, after: isValidDate(e.target.value) ? e.target.value : null })}
          />
        </label>
        {filterCount > 0 && (
          <button type="button" className="btn btn-secondary" onClick={() => onChange({ ...filters, videoId: null, after: null })}>
            Clear filters ({filterCount})
          </button>
        )}
      </div>

      <p id={hintId} className={styles.hint}>{hint}</p>
    </div>
  )
}
