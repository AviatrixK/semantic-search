import { useEffect, useState } from 'react'
import { streamUrls } from '../api/streams'
import { useInView } from '../hooks/useInView'
import { formatClock } from '../lib/format'
import { hueFor } from '../lib/videos'
import Icon from './Icon'
import styles from './VideoThumb.module.css'

interface Props {
  videoId: string
  /** The moment to show (seconds): a search result's thumbnail is the frame where the match starts. */
  at?: number
  duration?: number | null
}

/**
 * A thumbnail made from the video itself: a muted <video> that loads only its metadata and shows the frame at `at`
 * (no thumbnail files are stored). Until that frame is there, or if it cannot load, a colourful placeholder shows.
 * The file is only touched once the card is near the screen.
 */
export default function VideoThumb({ videoId, at = 0, duration }: Props) {
  const [ref, near] = useInView<HTMLDivElement>()
  const [url, setUrl] = useState<string | null>(null)
  const [shown, setShown] = useState(false)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (!near) return
    let alive = true
    streamUrls.get(videoId).then((u) => alive && setUrl(u), () => alive && setFailed(true))
    return () => {
      alive = false
    }
  }, [near, videoId])

  const hue = hueFor(videoId)
  const length = formatClock(duration)
  return (
    <div ref={ref} className={styles.thumb} style={{ background: `linear-gradient(135deg, hsl(${hue} 75% 84%), hsl(${(hue + 50) % 360} 75% 74%))` }}>
      <span className={styles.icon} aria-hidden="true"><Icon name="book" size={34} /></span>
      {url && !failed && (
        <video
          className={`${styles.frame} ${shown ? styles.shown : ''}`}
          src={`${url}#t=${Math.max(0.1, at).toFixed(1)}`}
          preload="metadata"
          muted
          playsInline
          tabIndex={-1}
          aria-hidden="true"
          disablePictureInPicture
          onLoadedData={() => setShown(true)}
          onError={() => setFailed(true)}
        />
      )}
      {length && <span className={styles.length}>{length}</span>}
    </div>
  )
}
