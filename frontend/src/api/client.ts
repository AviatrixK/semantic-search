import { ApiError, NetworkError, detailMessage, parseRetryAfter } from './errors.js'

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>
/** Runs `fn` while holding a lock shared by all tabs of this site (see webLocksRunner). */
export type LockRunner = <T>(fn: () => Promise<T>) => Promise<T>

export interface ClientOptions {
  baseUrl?: string
  fetch?: FetchLike
  runExclusive?: LockRunner
}

export interface RequestOptions {
  method?: string
  /** Serialised as JSON with the right Content-Type. */
  json?: unknown
  /** Raw body (e.g. FormData for uploads). */
  body?: BodyInit
  headers?: Record<string, string>
  signal?: AbortSignal
}

export interface ApiClient {
  getToken(): string | null
  setToken(token: string | null): void
  onTokenChange(cb: (token: string | null) => void): () => void
  /** Called when the session can no longer be refreshed: the app should show the login page. */
  onSessionExpired(cb: () => void): () => void
  /** Gets a new access token with the refresh cookie. At most one request in flight, across tabs. */
  refresh(): Promise<string | null>
  request<T = unknown>(path: string, options?: RequestOptions): Promise<T>
}

const AUTH_PATHS = ['/auth/login', '/auth/register', '/auth/refresh', '/auth/logout']
const noLock: LockRunner = (fn) => fn()

/**
 * Cross-tab mutex via the Web Locks API. The refresh token is single-use and presenting an already-used one
 * revokes ALL of the user's sessions. All tabs share one refresh cookie, so two tabs refreshing at the same moment
 * would both send the old token and trip that protection. Under the lock the second tab starts only after the first
 * has stored the rotated cookie, so it sends the new token. Falls back to no lock where the API is unavailable.
 */
export function webLocksRunner(name = 'svs-auth-refresh'): LockRunner {
  return (fn) => {
    const locks = (globalThis as { navigator?: { locks?: LockManager } }).navigator?.locks
    return locks ? (locks.request(name, fn) as Promise<Awaited<ReturnType<typeof fn>>>) : fn()
  }
}

export function createApiClient(options: ClientOptions = {}): ApiClient {
  const baseUrl = options.baseUrl ?? ''
  const doFetch: FetchLike = options.fetch ?? ((input, init) => fetch(input, init))
  const runExclusive = options.runExclusive ?? noLock

  let token: string | null = null
  let inflight: Promise<string | null> | null = null
  const tokenListeners = new Set<(t: string | null) => void>()
  const expiredListeners = new Set<() => void>()

  function setToken(next: string | null) {
    if (next === token) return
    token = next
    tokenListeners.forEach((cb) => cb(next))
  }

  function expireSession() {
    setToken(null)
    expiredListeners.forEach((cb) => cb())
  }

  async function send(path: string, o: RequestOptions, accessToken: string | null): Promise<Response> {
    const headers: Record<string, string> = { Accept: 'application/json', ...o.headers }
    if (accessToken) headers.Authorization = `Bearer ${accessToken}`
    let body = o.body
    if (o.json !== undefined) {
      headers['Content-Type'] = 'application/json'
      body = JSON.stringify(o.json)
    }
    try {
      return await doFetch(baseUrl + path, {
        method: o.method ?? (body === undefined ? 'GET' : 'POST'),
        headers,
        body,
        credentials: 'include',
        signal: o.signal,
      })
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e
      throw new NetworkError()
    }
  }

  async function toApiError(res: Response): Promise<ApiError> {
    let parsed: unknown
    try {
      parsed = await res.json()
    } catch {
      parsed = undefined
    }
    const message = detailMessage(parsed) ?? (res.statusText || `HTTP ${res.status}`)
    return new ApiError(res.status, message, parseRetryAfter(res.headers.get('Retry-After')))
  }

  async function doRefresh(): Promise<string | null> {
    const res = await send('/auth/refresh', { method: 'POST' }, null)
    if (res.ok) {
      const data = (await res.json()) as { access_token: string }
      setToken(data.access_token)
      return data.access_token
    }
    if (res.status === 401) {
      setToken(null)
      return null // no cookie, expired, or revoked: genuinely logged out
    }
    throw await toApiError(res) // 5xx etc.: the session may still be fine, so do not log the user out
  }

  function refresh(): Promise<string | null> {
    // Single flight: concurrent callers (parallel 401s, React StrictMode's double effect) share one request.
    if (!inflight) {
      inflight = runExclusive(doRefresh).finally(() => {
        inflight = null
      })
    }
    return inflight
  }

  async function request<T>(path: string, o: RequestOptions = {}): Promise<T> {
    const sentWith = token
    let res = await send(path, o, sentWith)

    if (res.status === 401 && !AUTH_PATHS.includes(path)) {
      // If another request refreshed while ours was in flight, just retry with the newer token.
      const fresh = token !== null && token !== sentWith ? token : await refresh()
      if (!fresh) {
        expireSession()
        throw new ApiError(401, 'Your session has expired. Please log in again.')
      }
      res = await send(path, o, fresh)
      if (res.status === 401) {
        expireSession()
        throw await toApiError(res)
      }
    }

    if (!res.ok) throw await toApiError(res)
    if (res.status === 204) return undefined as T
    return (res.headers.get('Content-Type') ?? '').includes('json') ? ((await res.json()) as T) : (undefined as T)
  }

  return {
    getToken: () => token,
    setToken,
    onTokenChange(cb) {
      tokenListeners.add(cb)
      return () => tokenListeners.delete(cb)
    },
    onSessionExpired(cb) {
      expiredListeners.add(cb)
      return () => expiredListeners.delete(cb)
    },
    refresh,
    request,
  }
}
