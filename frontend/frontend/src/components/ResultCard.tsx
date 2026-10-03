import type { SearchHit } from '../api'
import { formatDuration } from '../lib/format'
import { excerpt } from '../lib/highlight'
import styles from './ResultCard.module.css'

interface Props {
  /** DOM id: the search box points at it with aria-activedescendant. */
  id: string
  hit: SearchHit
  /** Highlighted by the arrow keys (Enter plays it). */
  active: boolean
  /** The one currently loaded in the player. */
  playing: boolean
  onPlay: () => void
}

/** One search result: video title, time range, the matching sentence in bold inside its chunk, and a subtle score bar. */
export default function ResultCard({ id, hit, active, playing, onPlay }: Props) {
  const { before, match, after } = excerpt(hit.text, hit.highlight)
  const pct = Math.round(Math.max(0, Math.min(1, hit.score)) * 100)
  return (
    <li
      id={id}
      role="option"
      aria-selected={active}
      className={`${styles.card} ${active ? styles.active : ''} ${playing ? styles.playing : ''}`}
      onClick={onPlay}
    >
      <div className={styles.head}>
        <span className={styles.title} title={hit.title}>{hit.title}</span>
        <span className={styles.range}>
          {formatDuration(hit.start_sec)} – {formatDuration(hit.end_sec)}
        </span>
        {playing && <span className={styles.badge}>Playing</span>}
      </div>
      <p className={styles.text}>
        {before}
        {match && <strong>{match}</strong>}
        {after}
      </p>
      <div
        className={styles.score}
        role="meter"
        aria-label="Similarity"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        aria-valuetext={`${pct}% match`}
        title={`Similarity ${hit.score.toFixed(2)}`}
      >
        <span style={{ width: `${pct}%` }} />
      </div>
    </li>
  )
}
