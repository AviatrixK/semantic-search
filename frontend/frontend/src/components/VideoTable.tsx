import type { Video } from '../api'
import { formatDate, formatDuration, statusBadge } from '../lib/format'
import styles from './VideoTable.module.css'

interface Props {
  videos: Video[]
  /** Ids with a reprocess/delete request in flight. */
  busy: ReadonlySet<string>
  onReprocess: (video: Video) => void
  onDelete: (video: Video) => void
}

export default function VideoTable({ videos, busy, onReprocess, onDelete }: Props) {
  return (
    <div className={styles.wrap}>
      <table className={styles.table}>
        <caption className="sr-only">Uploaded videos</caption>
        <thead>
          <tr>
            <th scope="col">Title</th>
            <th scope="col">Duration</th>
            <th scope="col">Status</th>
            <th scope="col">Created</th>
            <th scope="col"><span className="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {videos.map((v) => {
            const badge = statusBadge(v.status)
            const processing = v.status === 'uploaded'
            const locked = processing || busy.has(v.id) // the backend refuses to reprocess a running job (409)
            return (
              <tr key={v.id}>
                <td className={styles.title} title={v.title}>{v.title}</td>
                <td className={styles.num}>{formatDuration(v.duration_sec)}</td>
                <td><span className={`${styles.badge} ${styles[badge.tone]}`}>{badge.label}</span></td>
                <td className={styles.date}>{formatDate(v.created_at)}</td>
                <td className={styles.actions}>
                  <button type="button" className="btn btn-secondary" disabled={locked} onClick={() => onReprocess(v)}
                    aria-label={`Reprocess ${v.title}`}>
                    Reprocess
                  </button>
                  <button type="button" className={`btn btn-secondary ${styles.delete}`} disabled={locked} onClick={() => onDelete(v)}
                    aria-label={`Delete ${v.title}`}>
                    Delete
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
