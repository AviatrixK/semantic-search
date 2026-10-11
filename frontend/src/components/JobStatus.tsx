import type { Job } from '../api'
import { isTerminalStage, stageLabel } from '../lib/format'
import ProgressBar from './ProgressBar'
import styles from './JobStatus.module.css'

interface Props {
  /** Latest state from the live feed; undefined until the first event for this job arrives. */
  job: Job | undefined
  title: string
  onDismiss?: () => void
}

/** One card per processing job. Purely presentational: the numbers come from the live job feed (useJobFeed), never from polling. */
export default function JobStatus({ job, title, onDismiss }: Props) {
  const failed = job?.stage === 'failed'
  const done = job?.stage === 'done'
  const finished = job !== undefined && isTerminalStage(job.stage)
  const tone = failed ? 'bad' : done ? 'ok' : 'default'

  return (
    <article className={styles.card} aria-label={`Processing ${title}`}>
      <header className={styles.head}>
        <h3 className={styles.title} title={title}>{title}</h3>
        {finished && onDismiss && (
          <button type="button" className="btn btn-secondary" onClick={onDismiss}>Dismiss</button>
        )}
      </header>

      <div className={styles.stage} aria-live="polite">
        <span>{job ? stageLabel(job.stage) : 'Starting…'}</span>
        {job && !failed && <span className={styles.pct}>{job.progress}%</span>}
      </div>
      <ProgressBar value={job ? job.progress : null} label={`${title}: ${job ? stageLabel(job.stage) : 'starting'}`} tone={tone} />

      {failed && <p className={styles.error} role="alert">{job?.error || 'Processing failed.'}</p>}
      {done && <p className={styles.ok}>Ready. This video can now be searched.</p>}
    </article>
  )
}
