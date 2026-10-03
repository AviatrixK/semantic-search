import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { api } from '../api'
import type { TokenOut, User } from '../api'
import { ApiError, describeError } from '../api/errors'

export type AuthStatus = 'loading' | 'ready' | 'error'

interface AuthContextValue {
  user: User | null
  /** In memory only: never written to localStorage/sessionStorage. */
  accessToken: string | null
  status: AuthStatus
  bootstrapError: string | null
  login(email: string, password: string): Promise<void>
  register(email: string, password: string): Promise<void>
  logout(): Promise<void>
  retryBootstrap(): void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [accessToken, setAccessToken] = useState<string | null>(api.getToken())
  const [status, setStatus] = useState<AuthStatus>('loading')
  const [bootstrapError, setBootstrapError] = useState<string | null>(null)
  const run = useRef(0)

  useEffect(() => api.onTokenChange(setAccessToken), [])
  useEffect(() => api.onSessionExpired(() => setUser(null)), [])

  // On load: trade the refresh cookie for an access token, then fetch the user. React StrictMode runs this effect
  // twice in dev; both calls share the client's single in-flight /auth/refresh, so only one request is sent.
  const bootstrap = useCallback(async () => {
    const id = ++run.current
    const current = () => id === run.current
    setStatus('loading')
    setBootstrapError(null)
    try {
      const token = await api.refresh()
      const me = token ? await api.request<User>('/auth/me') : null
      if (!current()) return
      setUser(me)
      setStatus('ready')
    } catch (e) {
      if (!current()) return
      if (e instanceof ApiError && e.status === 401) {
        setUser(null) // the account behind the cookie is gone: treat as logged out
        setStatus('ready')
      } else {
        setBootstrapError(describeError(e))
        setStatus('error')
      }
    }
  }, [])

  useEffect(() => {
    void bootstrap()
  }, [bootstrap])

  const login = useCallback(async (email: string, password: string) => {
    const { access_token } = await api.request<TokenOut>('/auth/login', { method: 'POST', json: { email, password } })
    api.setToken(access_token)
    setUser(await api.request<User>('/auth/me'))
  }, [])

  const register = useCallback(
    async (email: string, password: string) => {
      await api.request('/auth/register', { method: 'POST', json: { email, password } })
      await login(email, password)
    },
    [login],
  )

  const logout = useCallback(async () => {
    try {
      await api.request('/auth/logout', { method: 'POST' })
    } catch {
      // Even if the server cannot be reached, forget the session locally.
    }
    api.setToken(null)
    setUser(null)
  }, [])

  const value = useMemo<AuthContextValue>(
    () => ({ user, accessToken, status, bootstrapError, login, register, logout, retryBootstrap: () => void bootstrap() }),
    [user, accessToken, status, bootstrapError, login, register, logout, bootstrap],
  )
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
