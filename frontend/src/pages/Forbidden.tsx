import { Link } from 'react-router-dom'
import styles from './page.module.css'

export default function Forbidden() {
  return (
    <main className={styles.page}>
      <h1>No access</h1>
      <p className={styles.muted}>Your account does not have permission to view this page.</p>
      <Link to="/">Back to home</Link>
    </main>
  )
}
