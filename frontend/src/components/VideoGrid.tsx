import { Link } from 'react-router-dom'
import type { Video } from '../api'
import { formatClock, formatDate, formatRelative } from '../lib/format'
import { watchPath } from '../lib/timeParam'
import VideoThumb from './VideoThumb'
import styles from './VideoGrid.module.css'

/** Responsive grid of video cards: thumbnail, two-line title, "length · uploaded 3 days ago". */
export default function VideoGrid({ videos }: { videos: Video[] }) {
  return (
    <ul className={styles.grid}>
      {videos.map((v) => (
        <li key={v.id}>
          <Link to={watchPath(v.id)} className={styles.card}>
            <VideoThumb videoId={v.id} duration={v.duration_sec} />
            <span className={styles.title} title={v.title}>{v.title}</span>
            <span className={styles.meta} title={formatDate(v.created_at)}>
              {[formatClock(v.duration_sec), `Uploaded ${formatRelative(v.created_at) || formatDate(v.created_at)}`].filter(Boolean).join(' · ')}
            </span>
          </Link>
        </li>
      ))}
    </ul>
  )
}
