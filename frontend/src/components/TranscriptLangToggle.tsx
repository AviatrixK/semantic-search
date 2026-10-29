import type { TranscriptLang } from '../lib/transcript'
import { TRANSCRIPT_LANGS } from '../lib/transcriptLang'
import styles from './TranscriptLangToggle.module.css'

interface Props {
  value: TranscriptLang
  onChange: (lang: TranscriptLang) => void
}

/** Original | English | Romanized: which version of the transcript is shown. Only rendered for non-English videos. */
export default function TranscriptLangToggle({ value, onChange }: Props) {
  return (
    <div className={styles.group} role="group" aria-label="Transcript language">
      {TRANSCRIPT_LANGS.map((l) => (
        <button
          key={l.value}
          type="button"
          className={`${styles.option} ${l.value === value ? styles.on : ''}`}
          aria-pressed={l.value === value}
          onClick={() => onChange(l.value)}
        >
          {l.label}
        </button>
      ))}
    </div>
  )
}
