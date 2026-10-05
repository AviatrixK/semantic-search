import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import Icon from './Icon'
import ThemeToggle from './ThemeToggle'
import styles from './AccountMenu.module.css'

/** The round avatar at the top right: opens a small menu with the account, the appearance switch and Log out. */
export default function AccountMenu() {
  const { user, logout } = useAuth()
  const [open, setOpen] = useState(false)
  const [loggingOut, setLoggingOut] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const { pathname } = useLocation()

  useEffect(() => setOpen(false), [pathname])

  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false)
        trigger.current?.focus()
      }
    }
    document.addEventListener('pointerdown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  if (!user) return null
  const initial = (user.email.trim()[0] ?? '?').toUpperCase()

  async function onLogout() {
    setLoggingOut(true)
    try {
      await logout()
    } finally {
      setLoggingOut(false)
    }
  }

  return (
    <div className={styles.root} ref={root}>
      <button ref={trigger} type="button" className={styles.avatar} onClick={() => setOpen((o) => !o)} aria-haspopup="menu" aria-expanded={open}
        aria-label={`Account menu for ${user.email}`}>
        {initial}
      </button>
      {open && (
        <div className={styles.menu} role="menu" aria-label="Account">
          <div className={styles.who}>
            <span className={styles.big}>{initial}</span>
            <span className={styles.lines}>
              <strong title={user.email}>{user.email}</strong>
              <span>{user.role === 'admin' ? 'Administrator' : 'Member'}</span>
            </span>
          </div>
          <ThemeToggle menu />
          <button type="button" className={styles.item} role="menuitem" onClick={onLogout} disabled={loggingOut}>
            <Icon name="logout" size={22} />
            <span>{loggingOut ? 'Logging out…' : 'Log out'}</span>
          </button>
        </div>
      )}
    </div>
  )
}
