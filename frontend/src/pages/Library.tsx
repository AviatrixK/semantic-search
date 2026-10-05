import { CardSkeletons } from '../components/Skeleton'
import VideoGrid from '../components/VideoGrid'
import { useVideos } from '../hooks/useVideos'
import styles from './Library.module.css'

export default function Library() {
  const { videos, loading, error, retry } = useVideos()
  const ready = videos?.filter((v) => v.status === 'ready') ?? []
  const processing = videos?.filter((v) => v.status === 'uploaded').length ?? 0

  return (
    <main className={styles.page}>
      <h1>Library</h1>

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

      {ready.length > 0 && (
        <>
          <VideoGrid videos={ready} />
          {processing > 0 && (
            <p className={styles.note}>{processing} more video{processing === 1 ? ' is' : 's are'} still being processed.</p>
          )}
        </>
      )}
    </main>
  )
}
