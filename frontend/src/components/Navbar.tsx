import { useState } from 'react'
import { Link, NavLink } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import styles from './Navbar.module.css'

export default function Navbar() {
  const { user, logout } = useAuth()
  const [loggingOut, setLoggingOut] = useState(false)

  async function onLogout() {
    setLoggingOut(true)
    try {
      await logout()
    } finally {
      setLoggingOut(false)
    }
  }

  const linkClass = ({ isActive }: { isActive: boolean }) => (isActive ? `${styles.link} ${styles.active}` : styles.link)

  return (
    <header className={styles.bar}>
      <nav className={styles.inner} aria-label="Main">
        <Link to="/" className={styles.brand}>
          Semantic Video Search
        </Link>
        {user ? (
          <>
            <div className={styles.links}>
              <NavLink to="/" end className={linkClass}>
                Search
              </NavLink>
              <NavLink to="/ask" className={linkClass}>
                Ask
              </NavLink>
              <NavLink to="/library" className={linkClass}>
                Library
              </NavLink>
              {user.role === 'admin' && (
                <NavLink to="/admin" className={linkClass}>
                  Admin
                </NavLink>
              )}
            </div>
            <div className={styles.account}>
              <span className={styles.email} title={user.email}>
                {user.email}
              </span>
              <span className={`${styles.role} ${user.role === 'admin' ? styles.admin : ''}`}>{user.role}</span>
              <button type="button" className="btn btn-secondary" onClick={onLogout} disabled={loggingOut}>
                {loggingOut ? 'Logging out…' : 'Log out'}
              </button>
            </div>
          </>
        ) : (
          <div className={styles.links}>
            <NavLink to="/login" className={linkClass}>
              Log in
            </NavLink>
            <NavLink to="/register" className={linkClass}>
              Register
            </NavLink>
          </div>
        )}
      </nav>
    </header>
  )
}
