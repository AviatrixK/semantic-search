import { useEffect, useRef } from 'react'
import type { Job } from '../api'
import { useJobPolling } from '../hooks/useJobPolling'
import { stageLabel } from '../lib/format'
import ProgressBar from './ProgressBar'
import styles from './JobStatus.module.css'

interface Props {
  jobId: string
  title: string
  /** Called once when the job reaches done or failed. */
  onFinished?: (job: Job) => void
  onDismiss?: () => void
}

export default function JobStatus({ jobId, title, onFinished, onDismiss }: Props) {
  const { job, problem, gaveUp, finished } = useJobPolling(jobId)
  const notified = useRef(false)

  useEffect(() => {
    if (finished && job && !notified.current) {
      notified.current = true
      onFinished?.(job)
    }
  }, [finished, job, onFinished])

  const failed = job?.stage === 'failed'
  const done = job?.stage === 'done'
  const progress = job ? job.progress : null // null: waiting for the first answer
  const tone = failed ? 'bad' : done ? 'ok' : 'default'

  return (
    <article className={styles.card} aria-label={`Processing ${title}`}>
      <header className={styles.head}>
        <h3 className={styles.title} title={title}>{title}</h3>
        {(finished || gaveUp) && onDismiss && (
          <button type="button" className="btn btn-secondary" onClick={onDismiss}>Dismiss</button>
        )}
      </header>

      <div className={styles.stage} aria-live="polite">
        <span>{job ? stageLabel(job.stage) : 'Starting…'}</span>
        {job && !failed && <span className={styles.pct}>{job.progress}%</span>}
      </div>
      <ProgressBar value={gaveUp ? 0 : progress} label={`${title}: ${job ? stageLabel(job.stage) : 'starting'}`} tone={tone} />

      {failed && <p className={styles.error} role="alert">{job?.error || 'Processing failed.'}</p>}
      {done && <p className={styles.ok}>Ready. This video can now be searched.</p>}
      {problem && !failed && (
        <p className={gaveUp ? styles.error : styles.warn} role={gaveUp ? 'alert' : 'status'}>
          {gaveUp ? problem : `${problem} Retrying…`}
        </p>
      )}
    </article>
  )
}
