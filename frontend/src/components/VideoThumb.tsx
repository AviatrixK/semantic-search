import { formatClock } from '../lib/format'
import { hueFor } from '../lib/videos'
import Icon from './Icon'
import styles from './VideoThumb.module.css'

interface Props {
  videoId: string
  duration?: number | null
}

/**
 * A colourful tile that stands in for a thumbnail. It deliberately touches no video file and makes no request: loading even
 * the metadata of every result cost megabytes per page and dozens of range requests. The video is fetched only when the user
 * opens it (see VideoPlayer). The colour is derived from the id so each video keeps its own.
 */
export default function VideoThumb({ videoId, duration }: Props) {
  const hue = hueFor(videoId)
  const length = formatClock(duration)
  return (
    <div className={styles.thumb} style={{ background: `linear-gradient(135deg, hsl(${hue} 75% 84%), hsl(${(hue + 50) % 360} 75% 74%))` }}>
      <span className={styles.icon} aria-hidden="true"><Icon name="book" size={34} /></span>
      {length && <span className={styles.length}>{length}</span>}
    </div>
  )
}
