import styles from './ProgressBar.module.css'

interface Props {
  /** 0-100, or null for "working, amount unknown". */
  value: number | null
  label: string
  tone?: 'default' | 'bad' | 'ok'
}

export default function ProgressBar({ value, label, tone = 'default' }: Props) {
  const pct = value === null ? null : Math.max(0, Math.min(100, Math.round(value)))
  return (
    <div
      className={styles.track}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct ?? undefined}
    >
      <div
        className={`${styles.bar} ${styles[tone]} ${pct === null ? styles.indeterminate : ''}`}
        style={pct === null ? undefined : { width: `${pct}%` }}
      />
    </div>
  )
}
