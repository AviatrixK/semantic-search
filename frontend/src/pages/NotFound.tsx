import { Link } from 'react-router-dom'
import styles from './page.module.css'

export default function NotFound() {
  return (
    <main className={styles.page}>
      <h1>Page not found</h1>
      <p className={styles.muted}>That address does not exist.</p>
      <Link to="/">Back to home</Link>
    </main>
  )
}
