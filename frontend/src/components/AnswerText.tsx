import { useMemo } from 'react'
import type { Citation } from '../api'
import { clock, segmentAnswer } from '../lib/answerText'
import styles from './AnswerText.module.css'

interface Props {
  text: string
  citations: Citation[]
  /** `${video_id}:${start_sec}` of the source currently loaded in the player. */
  selectedKey: string | null
  onCite: (citation: Citation) => void
}

export const citationKey = (c: Citation) => `${c.video_id}:${c.start_sec}`

/** An answer with its [n] markers turned into chips that play that moment. Rendered as text nodes only, never as HTML. */
export default function AnswerText({ text, citations, selectedKey, onCite }: Props) {
  const segments = useMemo(() => segmentAnswer(text, citations), [text, citations])
  return (
    <p className={styles.text}>
      {segments.map((s, i) =>
        s.type === 'text' ? (
          <span key={i}>{s.text}</span>
        ) : (
          <button
            key={i}
            type="button"
            className={`${styles.chip} ${selectedKey === citationKey(s.citation) ? styles.selected : ''}`}
            onClick={() => onCite(s.citation)}
            aria-label={`Play source ${s.citation.n}: ${s.citation.title} at ${clock(s.citation.start_sec)}`}
            title={`${s.citation.title} @ ${clock(s.citation.start_sec)}`}
          >
            {s.citation.n}
          </button>
        ),
      )}
    </p>
  )
}
