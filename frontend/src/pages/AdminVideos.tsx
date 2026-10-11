import { useMemo, useState } from 'react'
import { api } from '../api'
import type { UploadOut, Video } from '../api'
import { describeError } from '../api/errors'
import ConfirmDialog from '../components/ConfirmDialog'
import JobStatus from '../components/JobStatus'
import Spinner from '../components/Spinner'
import UploadCard from '../components/UploadCard'
import UploadForm from '../components/UploadForm'
import VideoTable from '../components/VideoTable'
import { useJobFeed } from '../hooks/useJobFeed'
import { useUploadQueue } from '../hooks/useUploadQueue'
import { useVideos } from '../hooks/useVideos'
import { hasActiveJob } from '../lib/jobFeed'
import styles from './AdminVideos.module.css'

const MAX_UPLOAD_MB = Number(import.meta.env.VITE_MAX_UPLOAD_MB) || 500 // keep equal to the backend's MAX_UPLOAD_MB

export default function AdminVideos() {
  const { videos, loading, error, reload, retry } = useVideos()
  const queue = useUploadQueue(reload)
  const [notice, setNotice] = useState<string | null>(null)
  const [busy, setBusy] = useState<ReadonlySet<string>>(new Set())
  const [toDelete, setToDelete] = useState<Video | null>(null)
  const [deleting, setDeleting] = useState(false)

  // The live feed is open only while something is being processed: a video that is not "ready"/"failed" yet, a card waiting for
  // its first event, or a job the feed reports as running. When the last one finishes this turns false and the connection closes.
  const [watching, setWatching] = useState(false) // jobs this page started (uploads/reprocess) whose first event may still be on its way
  const feed = useJobFeed(watching, { onFinished: () => void reload(), onResync: () => void reload() })
  const processingVideos = videos?.some((v) => v.status === 'uploaded') ?? false
  const waitingForFirstEvent = queue.items.some((i) => i.phase === 'processing' && i.jobId !== null && !(i.jobId in feed.jobs))
  const active = processingVideos || waitingForFirstEvent || hasActiveJob(feed.jobs)
  if (active !== watching) setWatching(active) // derived state: adjusting during render is the React-recommended form

  // Jobs the feed knows about that have no upload card on this page (e.g. after a refresh): shown with the video's title.
  const trackedJobIds = useMemo(() => new Set(queue.items.map((i) => i.jobId).filter(Boolean)), [queue.items])
  const titleOf = (videoId: string) => videos?.find((v) => v.id === videoId)?.title ?? 'Video'
  const otherJobs = Object.values(feed.jobs).filter((j) => !trackedJobIds.has(j.id))

  const setBusyFor = (id: string, on: boolean) =>
    setBusy((cur) => {
      const next = new Set(cur)
      if (on) next.add(id)
      else next.delete(id)
      return next
    })

  async function reprocess(video: Video) {
    setNotice(null)
    setBusyFor(video.id, true)
    try {
      const res = await api.request<UploadOut>(`/api/videos/${video.id}/reprocess`, { method: 'POST' })
      queue.track(res.video_id, res.job_id, video.title)
    } catch (e) {
      setNotice(describeError(e, 'reprocess'))
    } finally {
      setBusyFor(video.id, false)
      void reload()
    }
  }

  async function confirmDelete() {
    if (!toDelete) return
    setDeleting(true)
    setNotice(null)
    try {
      await api.request(`/api/videos/${toDelete.id}`, { method: 'DELETE' })
      queue.dismissByVideo(toDelete.id)
    } catch (e) {
      setNotice(describeError(e, 'delete'))
    } finally {
      setDeleting(false)
      setToDelete(null)
      void reload()
    }
  }

  return (
    <main className={styles.page}>
      <h1>Videos</h1>

      {notice && (
        <div className={styles.notice} role="alert">
          <span>{notice}</span>
          <button type="button" className="btn btn-secondary" onClick={() => setNotice(null)}>Dismiss</button>
        </div>
      )}

      <section aria-labelledby="upload-heading" className={styles.section}>
        <h2 id="upload-heading">Upload</h2>
        <UploadForm maxMb={MAX_UPLOAD_MB} onSubmit={queue.add} />
      </section>

      {(queue.items.length > 0 || otherJobs.length > 0) && (
        <section aria-labelledby="jobs-heading" className={styles.section}>
          <h2 id="jobs-heading">Uploads and processing</h2>
          {feed.status === 'reconnecting' && (
            <p className={styles.empty} role="status">Live updates were interrupted. Reconnecting…</p>
          )}
          {feed.status === 'stopped' && (
            <p className={styles.empty} role="status">Live updates are off for this session. Reload the page to see progress.</p>
          )}
          <ul className={styles.cards}>
            {queue.items.map((item) => (
              <li key={item.id}>
                <UploadCard
                  item={item}
                  job={item.jobId ? feed.jobs[item.jobId] : undefined}
                  onCancel={queue.cancel}
                  onDismiss={(id) => {
                    if (item.jobId) feed.dismiss(item.jobId)
                    queue.dismiss(id)
                  }}
                />
              </li>
            ))}
            {otherJobs.map((job) => (
              <li key={job.id}>
                <JobStatus job={job} title={titleOf(job.video_id)} onDismiss={() => feed.dismiss(job.id)} />
              </li>
            ))}
          </ul>
        </section>
      )}

      <section aria-labelledby="library-heading" className={styles.section}>
        <h2 id="library-heading">Library</h2>
        {videos === null && loading && <Spinner label="Loading videos…" />}
        {error && (
          <div className={styles.notice} role="alert">
            <span>{error}</span>
            <button type="button" className="btn btn-secondary" onClick={() => void retry()}>Try again</button>
          </div>
        )}
        {videos !== null && videos.length === 0 && <p className={styles.empty}>No videos yet. Upload one above.</p>}
        {videos !== null && videos.length > 0 && (
          <VideoTable videos={videos} busy={busy} onReprocess={(v) => void reprocess(v)} onDelete={setToDelete} />
        )}
      </section>

      <ConfirmDialog
        open={toDelete !== null}
        title="Delete this video?"
        message={`"${toDelete?.title ?? ''}" and all of its transcript chunks will be permanently removed.`}
        confirmLabel="Delete"
        busy={deleting}
        onConfirm={() => void confirmDelete()}
        onCancel={() => setToDelete(null)}
      />
    </main>
  )
}
