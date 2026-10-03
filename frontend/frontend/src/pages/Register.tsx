import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { ApiError, describeError, rateLimitMessage } from '../api/errors'
import { useAuth } from '../context/AuthContext'
import { useCountdown } from '../hooks/useCountdown'
import { checkPassword, passwordIsValid } from '../lib/password'
import styles from './auth.module.css'

const DEFAULT_WAIT_SEC = 60

export default function Register() {
  const { user, register } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [touched, setTouched] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [blockedUntil, setBlockedUntil] = useState<number | null>(null)
  const secondsLeft = useCountdown(blockedUntil)
  const blocked = secondsLeft > 0

  useEffect(() => {
    if (blockedUntil !== null && secondsLeft === 0) setBlockedUntil(null)
  }, [blockedUntil, secondsLeft])

  if (user) return <Navigate to="/" replace />

  const checks = checkPassword(password)
  const valid = passwordIsValid(password)

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    if (submitting || blocked) return
    setTouched(true)
    if (!valid) return
    setError(null)
    setSubmitting(true)
    try {
      await register(email.trim(), password) // registers, then logs in
    } catch (err) {
      if (err instanceof ApiError && err.status === 429) {
        setBlockedUntil(Date.now() + (err.retryAfter ?? DEFAULT_WAIT_SEC) * 1000)
      } else {
        setError(describeError(err, 'register'))
      }
    } finally {
      setSubmitting(false)
    }
  }

  const message = blocked ? rateLimitMessage(secondsLeft) : error

  return (
    <main className={styles.page}>
      <form className={styles.card} onSubmit={onSubmit} aria-busy={submitting}>
        <h1>Create account</h1>
        <p className={styles.lead}>Register to search videos by meaning.</p>

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
            autoComplete="new-password"
            required
            aria-describedby="password-rules"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={submitting}
          />
        </label>
        <ul id="password-rules" className={styles.rules}>
          {checks.map((c) => (
            <li key={c.label} className={c.ok ? styles.ok : touched ? styles.bad : undefined}>
              <span aria-hidden="true">{c.ok ? '✓' : '•'}</span> {c.label}
              <span className="sr-only">{c.ok ? ' (met)' : ' (not met)'}</span>
            </li>
          ))}
        </ul>

        <button className="btn btn-primary" type="submit" disabled={submitting || blocked}>
          {submitting ? 'Creating account…' : blocked ? `Try again in ${secondsLeft}s` : 'Create account'}
        </button>

        <p className={styles.alt}>
          Already registered? <Link to="/login">Log in</Link>
        </p>
      </form>
    </main>
  )
}
