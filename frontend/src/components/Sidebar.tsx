import { NavLink } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { preloadRoute } from '../routes'
import Icon from './Icon'
import styles from './Sidebar.module.css'

const ITEMS = [
  { to: '/', label: 'Home', icon: 'home', end: true },
  { to: '/ask', label: 'Ask', icon: 'ask' },
  { to: '/library', label: 'Library', icon: 'library' },
] as const

/** Left navigation: full width, a slim icon rail (collapsed or on tablets), or a bottom bar on phones. */
export default function Sidebar({ collapsed }: { collapsed: boolean }) {
  const { user } = useAuth()
  if (!user) return null
  const linkClass = ({ isActive }: { isActive: boolean }) => (isActive ? `${styles.item} ${styles.active}` : styles.item)
  // Start downloading a page as soon as the pointer or keyboard reaches its link.
  const warm = (to: string) => ({ onPointerEnter: () => preloadRoute(to), onFocus: () => preloadRoute(to) })

  return (
    <nav className={styles.side} data-collapsed={collapsed ? '' : undefined} aria-label="Main">
      {ITEMS.map((i) => (
        <NavLink key={i.to} to={i.to} end={'end' in i && i.end} className={linkClass} {...warm(i.to)}>
          <Icon name={i.icon} />
          <span>{i.label}</span>
        </NavLink>
      ))}
      {user.role === 'admin' && (
        <NavLink to="/admin" className={linkClass} {...warm('/admin')}>
          <Icon name="admin" />
          <span>Admin</span>
        </NavLink>
      )}
    </nav>
  )
}
