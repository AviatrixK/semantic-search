import { useMemo, useState } from 'react'
import { Navigate, useSearchParams } from 'react-router-dom'
import { CardSkeletons } from '../components/Skeleton'
import VideoGrid from '../components/VideoGrid'
import { useVideos } from '../hooks/useVideos'
import { parseFilters, searchPagePath } from '../lib/searchParams'
import { SORT_OPTIONS, sortVideos } from '../lib/videos'
import type { VideoSort } from '../lib/videos'
import styles from './Home.module.css'

/** The front page: every searchable video as a thumbnail grid, with sort chips on top. */
export default function Home() {
  const [params] = useSearchParams()
  const [sort, setSort] = useState<VideoSort>('newest')
  const { videos, loading, error, retry } = useVideos()
  const ready = useMemo(() => sortVideos(videos?.filter((v) => v.status === 'ready') ?? [], sort), [videos, sort])
  const processing = videos?.filter((v) => v.status === 'uploaded').length ?? 0

  // Old links looked like /?q=pauses: send them to the results page.
  const legacy = params.get('q')
  if (legacy && legacy.trim()) return <Navigate to={searchPagePath(legacy, parseFilters(params))} replace />

  return (
    <main className={styles.page}>
      <h1 className="sr-only">Home</h1>
      <div className={styles.chips} role="group" aria-label="Sort videos">
        {SORT_OPTIONS.map((o) => (
          <button key={o.value} type="button" className={`${styles.chip} ${sort === o.value ? styles.on : ''}`} aria-pressed={sort === o.value}
            onClick={() => setSort(o.value)}>
            {o.label}
          </button>
        ))}
      </div>

      {videos === null && loading && <CardSkeletons label="Loading videos…" />}

      {error && (
        <div className={styles.error} role="alert">
          <span>{error}</span>
          <button type="button" className="btn btn-secondary" onClick={() => void retry()}>Try again</button>
        </div>
      )}

      {videos !== null && ready.length === 0 && (
        <p className={styles.empty}>
          {processing > 0
            ? `${processing} video${processing === 1 ? ' is' : 's are'} still being processed. They will appear here when ready.`
            : 'There are no videos yet. An administrator can upload some from the Admin page.'}
        </p>
      )}

      {ready.length > 0 && <VideoGrid videos={ready} />}
      {ready.length > 0 && processing > 0 && (
        <p className={styles.note}>{processing} more video{processing === 1 ? ' is' : 's are'} still being processed.</p>
      )}
    </main>
  )
}
