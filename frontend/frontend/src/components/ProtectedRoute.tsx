import type { ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import type { User } from '../api'
import Forbidden from '../pages/Forbidden'
import Spinner from './Spinner'

interface Props {
  /** Only users with this role may see the page; others get a 403 page (logged out users go to /login). */
  role?: User['role']
  children: ReactNode
}

export default function ProtectedRoute({ role, children }: Props) {
  const { user, status } = useAuth()
  const location = useLocation()

  if (status === 'loading') return <Spinner label="Checking your session…" />
  if (!user) return <Navigate to="/login" replace state={{ from: location }} />
  if (role && user.role !== role) return <Forbidden />
  return <>{children}</>
}
