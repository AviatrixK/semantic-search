import type { Job } from '../api'
import { formatBytes } from '../lib/format'
import type { QueueItem } from '../lib/uploadQueue'
import JobStatus from './JobStatus'
import ProgressBar from './ProgressBar'
import styles from './JobStatus.module.css'

interface Props {
  item: QueueItem
  onCancel: (id: string) => void
  onDismiss: (id: string) => void
  onJobFinished: (job: Job) => void
}

/** One card per queued upload: waiting -> uploading (bytes) -> processing (live job status) or error. */
export default function UploadCard({ item, onCancel, onDismiss, onJobFinished }: Props) {
  if (item.phase === 'processing' && item.jobId) {
    return <JobStatus jobId={item.jobId} title={item.title} onFinished={onJobFinished} onDismiss={() => onDismiss(item.id)} />
  }

  const pct = item.total > 0 ? Math.round((item.loaded / item.total) * 100) : 0
  const sending = item.phase === 'uploading'
  const finishing = sending && pct >= 100 // all bytes sent, waiting for the server to store the file

  return (
    <article className={styles.card} aria-label={`Upload ${item.title}`}>
      <header className={styles.head}>
        <h3 className={styles.title} title={item.title}>{item.title}</h3>
        {item.phase === 'error' ? (
          <button type="button" className="btn btn-secondary" onClick={() => onDismiss(item.id)}>Dismiss</button>
        ) : (
          <button type="button" className="btn btn-secondary" onClick={() => onCancel(item.id)}>Cancel</button>
        )}
      </header>

      {item.phase === 'queued' && <p className={styles.warn}>Waiting to upload ({formatBytes(item.fileSize)})…</p>}
      {sending && (
        <>
          <div className={styles.stage} aria-live="polite">
            <span>{finishing ? 'Finishing upload…' : 'Uploading'}</span>
            <span className={styles.pct}>
              {pct}% · {formatBytes(item.loaded)} of {formatBytes(item.total)}
            </span>
          </div>
          <ProgressBar value={finishing ? null : pct} label={`Uploading ${item.title}`} />
        </>
      )}
      {item.phase === 'error' && <p className={styles.error} role="alert">{item.error}</p>}
    </article>
  )
}
