import { useEffect, useRef, useState } from 'react'
import { formatDuration } from '../lib/format'
import { scrollTargetFor } from '../lib/listNav'
import type { TranscriptLine } from '../lib/transcript'
import styles from './TranscriptPanel.module.css'

interface Props {
  lines: TranscriptLine[]
  /** Line being spoken now (-1: none yet). */
  activeIndex: number
  onSelect: (line: TranscriptLine) => void
}

/**
 * The transcript as clickable lines. The current line is highlighted and kept in view; scrolling it by hand pauses that
 * until "Follow playback" (or a click on a line) turns it back on.
 */
export default function TranscriptPanel({ lines, activeIndex, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const items = useRef<(HTMLLIElement | null)[]>([])
  const [follow, setFollow] = useState(true)

  useEffect(() => {
    if (!follow || activeIndex < 0) return
    const container = box.current
    const el = items.current[activeIndex]
    if (!container || !el) return
    // Scroll the panel itself, never the page (scrollIntoView would drag the whole window along).
    const target = scrollTargetFor(container.scrollTop, container.clientHeight, el.offsetTop, el.offsetHeight)
    if (target !== null) {
      const calm = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
      container.scrollTo({ top: target, behavior: calm ? 'auto' : 'smooth' })
    }
  }, [activeIndex, follow])

  const pause = () => setFollow(false)

  return (
    <div className={styles.wrap}>
      {!follow && (
        <button type="button" className={`btn btn-secondary ${styles.follow}`} onClick={() => setFollow(true)}>
          Follow playback
        </button>
      )}
      <div
        ref={box}
        className={styles.box}
        onWheel={pause}
        onTouchMove={pause}
        onKeyDown={(e) => {
          if (['PageUp', 'PageDown', 'Home', 'End', 'ArrowUp', 'ArrowDown'].includes(e.key)) pause()
        }}
      >
        <ol className={styles.list}>
          {lines.map((line, i) => (
            <li key={line.id} ref={(el) => { items.current[i] = el }}>
              <button
                type="button"
                className={`${styles.line} ${i === activeIndex ? styles.active : ''}`}
                aria-current={i === activeIndex ? 'true' : undefined}
                onClick={() => {
                  setFollow(true)
                  onSelect(line)
                }}
              >
                <time className={styles.time}>{formatDuration(line.start)}</time>
                <span>{line.text}</span>
              </button>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}
