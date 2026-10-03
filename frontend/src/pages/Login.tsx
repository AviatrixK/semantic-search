import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, Navigate, useLocation } from 'react-router-dom'
import { ApiError, describeError, rateLimitMessage } from '../api/errors'
import { useAuth } from '../context/AuthContext'
import { useCountdown } from '../hooks/useCountdown'
import styles from './auth.module.css'

const DEFAULT_WAIT_SEC = 60 // the backend's rate-limit window, used if Retry-After is missing

/** Where to go after logging in: the page the user was sent away from, if it is a local path. */
function destination(state: unknown): string {
  const from = (state as { from?: { pathname?: string; search?: string } } | null)?.from
  return from?.pathname && from.pathname.startsWith('/') && !from.pathname.startsWith('//')
    ? `${from.pathname}${from.search ?? ''}`
    : '/'
}

export default function Login() {
  const { user, login } = useAuth()
  const location = useLocation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [blockedUntil, setBlockedUntil] = useState<number | null>(null)
  const secondsLeft = useCountdown(blockedUntil)
  const blocked = secondsLeft > 0

  useEffect(() => {
    if (blockedUntil !== null && secondsLeft === 0) setBlockedUntil(null) // wait is over: allow trying again
  }, [blockedUntil, secondsLeft])

  if (user) return <Navigate to={destination(location.state)} replace />

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    if (submitting || blocked) return
    setError(null)
    setSubmitting(true)
    try {
      await login(email.trim(), password)
    } catch (err) {
      if (err instanceof ApiError && err.status === 429) {
        setBlockedUntil(Date.now() + (err.retryAfter ?? DEFAULT_WAIT_SEC) * 1000)
      } else {
        setError(describeError(err, 'login'))
      }
    } finally {
      setSubmitting(false)
    }
  }

  const message = blocked ? rateLimitMessage(secondsLeft) : error

  return (
    <main className={styles.page}>
      <form className={styles.card} onSubmit={onSubmit} aria-busy={submitting}>
        <h1>Log in</h1>
        <p className={styles.lead}>Welcome back. Log in to search your videos.</p>

        <div className={styles.error} role="alert" hidden={!message}>
          {message}
        </div>

        <label className={styles.field}>
          <span>Email</span>
          <input
            className="input"
            type="email"
            name="email"
            autoComplete="username"
            required
            autoFocus
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={submitting}
          />
        </label>
        <label className={styles.field}>
          <span>Password</span>
          <input
            className="input"
            type="password"
            name="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={submitting}
          />
        </label>

        <button className="btn btn-primary" type="submit" disabled={submitting || blocked}>
          {submitting ? 'Logging in…' : blocked ? `Try again in ${secondsLeft}s` : 'Log in'}
        </button>

        <p className={styles.alt}>
          No account yet? <Link to="/register">Create one</Link>
        </p>
      </form>
    </main>
  )
}
