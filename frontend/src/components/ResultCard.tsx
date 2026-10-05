import { Link } from 'react-router-dom'
import type { SearchHit, Video } from '../api'
import { formatClock, formatDate, formatRelative } from '../lib/format'
import { excerpt } from '../lib/highlight'
import { foundByBadge, scoreText } from '../lib/searchParams'
import type { SearchMode } from '../lib/searchParams'
import { watchPath } from '../lib/timeParam'
import VideoThumb from './VideoThumb'
import styles from './ResultCard.module.css'

interface Props {
  hit: SearchHit
  /** The hit's video, when known (gives the length and the upload date). */
  video?: Video
  /** The mode this result came from: names the score bar (keyword scores are not similarities). */
  mode: SearchMode
}

/**
 * One search result, YouTube style: the thumbnail is the frame where the match starts, then the title, upload info, a
 * "jump to" time chip and the matching sentence in bold. The whole row opens the video at that moment.
 */
export default function ResultCard({ hit, video, mode }: Props) {
  const { before, match, after } = excerpt(hit.text, hit.highlight)
  const pct = Math.round(Math.max(0, Math.min(1, hit.score)) * 100)
  const badge = mode === 'hybrid' ? foundByBadge(hit.found_by) : null
  const score = scoreText(mode, hit.score)
  const uploaded = video ? formatRelative(video.created_at) || formatDate(video.created_at) : ''
  return (
    <Link to={watchPath(hit.video_id, hit.start_sec)} className={styles.row}>
      <div className={styles.thumb}>
        <VideoThumb videoId={hit.video_id} at={hit.start_sec} duration={video?.duration_sec} />
      </div>
      <div className={styles.body}>
        <h2 className={styles.title}>{hit.title}</h2>
        <p className={styles.meta}>{[uploaded && `Uploaded ${uploaded}`, video && formatClock(video.duration_sec)].filter(Boolean).join(' · ')}</p>
        <p className={styles.moment}>
          <span className={styles.chip}>Jump to {formatClock(hit.start_sec)}</span>
          <span className={styles.range}>{formatClock(hit.start_sec)} – {formatClock(hit.end_sec)}</span>
        </p>
        <p className={styles.text}>
          {before}
          {match && <strong>{match}</strong>}
          {after}
        </p>
        <div className={styles.foot}>
          {badge && <span className={styles.found} title={badge.title}>{badge.label}</span>}
          <span
            className={styles.score}
            role="meter"
            aria-label={score.label}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct}
            aria-valuetext={`${pct}% match`}
            title={score.title}
          >
            <span style={{ width: `${pct}%` }} />
          </span>
        </div>
      </div>
    </Link>
  )
}
