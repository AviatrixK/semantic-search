import { useCallback, useState } from 'react'
import { api } from '../api'
import type { UploadOut, Video } from '../api'
import { describeError } from '../api/errors'
import ConfirmDialog from '../components/ConfirmDialog'
import Spinner from '../components/Spinner'
import UploadCard from '../components/UploadCard'
import UploadForm from '../components/UploadForm'
import VideoTable from '../components/VideoTable'
import { useUploadQueue } from '../hooks/useUploadQueue'
import { useVideos } from '../hooks/useVideos'
import styles from './AdminVideos.module.css'

const MAX_UPLOAD_MB = Number(import.meta.env.VITE_MAX_UPLOAD_MB) || 500 // keep equal to the backend's MAX_UPLOAD_MB

export default function AdminVideos() {
  const { videos, loading, error, reload, retry } = useVideos()
  const queue = useUploadQueue(reload)
  const [notice, setNotice] = useState<string | null>(null)
  const [busy, setBusy] = useState<ReadonlySet<string>>(new Set())
  const [toDelete, setToDelete] = useState<Video | null>(null)
  const [deleting, setDeleting] = useState(false)

  const setBusyFor = (id: string, on: boolean) =>
    setBusy((cur) => {
      const next = new Set(cur)
      if (on) next.add(id)
      else next.delete(id)
      return next
    })

  const onJobFinished = useCallback(() => void reload(), [reload]) // refresh the table when a card reaches done/failed

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

      {queue.items.length > 0 && (
        <section aria-labelledby="jobs-heading" className={styles.section}>
          <h2 id="jobs-heading">Uploads and processing</h2>
          <ul className={styles.cards}>
            {queue.items.map((item) => (
              <li key={item.id}>
                <UploadCard item={item} onCancel={queue.cancel} onDismiss={queue.dismiss} onJobFinished={onJobFinished} />
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
