import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, NavLink, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useSlashFocus } from '../hooks/useSlashFocus'
import { DEFAULT_FILTERS, parseFilters, searchPagePath } from '../lib/searchParams'
import AccountMenu from './AccountMenu'
import Icon from './Icon'
import ThemeToggle from './ThemeToggle'
import styles from './Header.module.css'

/** The search box in the middle of the header. Enter opens the results page; on that page the box mirrors the query. */
function HeaderSearch() {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const [params] = useSearchParams()
  const onResults = pathname === '/search'
  const urlQuery = params.get('q') ?? ''
  const [value, setValue] = useState(onResults ? urlQuery : '')
  const input = useRef<HTMLInputElement>(null)
  useSlashFocus(input)

  useEffect(() => {
    if (onResults) setValue(urlQuery) // back/forward and shared links keep the box in step
  }, [onResults, urlQuery])

  function submit(e: FormEvent) {
    e.preventDefault()
    const q = value.trim()
    if (q.length < 2) {
      input.current?.focus()
      return
    }
    navigate(searchPagePath(q, onResults ? parseFilters(params) : DEFAULT_FILTERS)) // keep mode and filters while searching again
  }

  return (
    <form role="search" className={styles.search} onSubmit={submit}>
      <input
        ref={input}
        className={styles.box}
        type="search"
        name="q"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Search the videos  ( / )"
        aria-label="Search videos"
        autoComplete="off"
        enterKeyHint="search"
        maxLength={500}
      />
      <button type="submit" className={styles.go} aria-label="Search">
        <Icon name="search" size={22} />
      </button>
    </form>
  )
}

interface Props {
  /** The menu button collapses the sidebar (hidden while there is no sidebar). */
  onToggleSidebar?: () => void
}

/** YouTube-style top bar: menu button and logo on the left, search in the middle, account on the right. */
export default function Header({ onToggleSidebar }: Props) {
  const { user } = useAuth()
  return (
    <header className={styles.bar}>
      <div className={styles.left}>
        {user && onToggleSidebar && (
          <button type="button" className={styles.menuBtn} onClick={onToggleSidebar} aria-label="Show or hide the menu">
            <Icon name="menu" />
          </button>
        )}
        <Link to="/" className={styles.brand} aria-label="Semantic Video Search, home">
          <span className={styles.mark} aria-hidden="true"><Icon name="cap" size={20} /></span>
          <span className={styles.name}>Semantic Video Search</span>
        </Link>
      </div>
      <div className={styles.center}>{user && <HeaderSearch />}</div>
      <div className={styles.right}>
        {user ? (
          <AccountMenu />
        ) : (
          <>
            <ThemeToggle />
            <NavLink to="/login" className={styles.link}>Log in</NavLink>
            <NavLink to="/register" className={`btn btn-primary ${styles.register}`}>Register</NavLink>
          </>
        )}
      </div>
    </header>
  )
}
