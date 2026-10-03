import styles from './Spinner.module.css'

export default function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className={styles.wrap} role="status" aria-live="polite">
      <span className={styles.ring} aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}
