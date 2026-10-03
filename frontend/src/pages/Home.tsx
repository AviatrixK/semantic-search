import { useAuth } from '../context/AuthContext'
import styles from './page.module.css'

export default function Home() {
  const { user } = useAuth()
  return (
    <main className={styles.page}>
      <h1>Welcome{user ? `, ${user.email}` : ''}</h1>
      <p className={styles.muted}>
        You are logged in as <strong>{user?.role}</strong>. Search by meaning arrives in a later phase; this page is a
        placeholder.
      </p>
    </main>
  )
}
