import styles from './Skeleton.module.css'

/** Placeholder shapes shown while content loads (the look of the finished thing, so the page does not jump). */
export function ResultSkeletons({ count = 3, label = 'Searching…' }: { count?: number; label?: string }) {
  return (
    <div role="status" aria-live="polite" className={styles.stack}>
      <span className="sr-only">{label}</span>
      {Array.from({ length: count }, (_, i) => (
        <div key={i} aria-hidden="true" className={styles.card}>
          <span className="skeleton" style={{ height: '1rem', width: `${55 + ((i * 17) % 30)}%` }} />
          <span className="skeleton" style={{ height: '0.8rem', width: '100%' }} />
          <span className="skeleton" style={{ height: '0.8rem', width: `${70 - ((i * 11) % 25)}%` }} />
        </div>
      ))}
    </div>
  )
}

export function CardSkeletons({ count = 6, label = 'Loading…' }: { count?: number; label?: string }) {
  return (
    <div role="status" aria-live="polite" className={styles.grid}>
      <span className="sr-only">{label}</span>
      {Array.from({ length: count }, (_, i) => (
        <div key={i} aria-hidden="true" className={styles.card}>
          <span className="skeleton" style={{ aspectRatio: '16 / 9', width: '100%' }} />
          <span className="skeleton" style={{ height: '1rem', width: '80%' }} />
          <span className="skeleton" style={{ height: '0.8rem', width: '45%' }} />
        </div>
      ))}
    </div>
  )
}
