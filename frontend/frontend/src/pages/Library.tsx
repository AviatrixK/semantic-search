import { Link } from 'react-router-dom'
import Spinner from '../components/Spinner'
import { useVideos } from '../hooks/useVideos'
import { formatDate, formatDuration } from '../lib/format'
import { watchPath } from '../lib/timeParam'
import styles from './Library.module.css'

export default function Library() {
  const { videos, loading, error, retry } = useVideos()
  const ready = videos?.filter((v) => v.status === 'ready') ?? []
  const processing = videos?.filter((v) => v.status === 'uploaded').length ?? 0

  return (
    <main className={styles.page}>
      <h1>Library</h1>

      {videos === null && loading && <Spinner label="Loading videos…" />}

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

      {ready.length > 0 && (
        <>
          <ul className={styles.grid}>
            {ready.map((v) => (
              <li key={v.id}>
                <Link to={watchPath(v.id)} className={styles.card}>
                  <span className={styles.thumb} aria-hidden="true">▶</span>
                  <span className={styles.title} title={v.title}>{v.title}</span>
                  <span className={styles.meta}>
                    <span>{formatDuration(v.duration_sec)}</span>
                    <span>{formatDate(v.created_at)}</span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
          {processing > 0 && (
            <p className={styles.note}>{processing} more video{processing === 1 ? ' is' : 's are'} still being processed.</p>
          )}
        </>
      )}
    </main>
  )
}
