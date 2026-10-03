import { ApiError, NetworkError, detailMessage, parseRetryAfter } from './errors.js'

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>
/** Runs `fn` while holding a lock shared by all tabs of this site (see webLocksRunner). */
export type LockRunner = <T>(fn: () => Promise<T>) => Promise<T>

/** The slice of XMLHttpRequest the upload code uses (so tests can supply a fake). */
export interface XhrLike {
  open(method: string, url: string): void
  setRequestHeader(name: string, value: string): void
  send(body: FormData): void
  abort(): void
  getAllResponseHeaders(): string
  withCredentials: boolean
  status: number
  responseText: string
  upload: { onprogress: ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null }
  onload: (() => void) | null
  onerror: (() => void) | null
  ontimeout: (() => void) | null
  onabort: (() => void) | null
}

export interface ClientOptions {
  baseUrl?: string
  fetch?: FetchLike
  runExclusive?: LockRunner
  xhr?: () => XhrLike
}

export interface RequestOptions {
  method?: string
  /** Serialised as JSON with the right Content-Type. */
  json?: unknown
  /** Raw body (e.g. FormData). */
  body?: BodyInit
  headers?: Record<string, string>
  signal?: AbortSignal
}

export interface UploadOptions {
  /** Bytes sent so far / total. Called again from zero if the upload is retried after a token refresh. */
  onProgress?: (loaded: number, total: number) => void
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
  /** POST a FormData with upload progress (fetch cannot report upload progress, XMLHttpRequest can). */
  upload<T = unknown>(path: string, form: FormData, options?: UploadOptions): Promise<T>
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

function abortError(): DOMException {
  return new DOMException('Aborted', 'AbortError')
}

/** Builds a fetch-style Response from a finished XHR so uploads share the error/parse code with request(). */
function xhrToResponse(xhr: XhrLike): Response {
  const headers = new Headers()
  for (const line of xhr.getAllResponseHeaders().trim().split(/[\r\n]+/)) {
    const i = line.indexOf(':')
    if (i > 0) headers.append(line.slice(0, i).trim(), line.slice(i + 1).trim())
  }
  const noBody = xhr.status === 204 || xhr.status === 205 || xhr.status === 304
  return new Response(noBody ? null : xhr.responseText, { status: xhr.status, headers })
}

export function createApiClient(options: ClientOptions = {}): ApiClient {
  const baseUrl = options.baseUrl ?? ''
  const doFetch: FetchLike = options.fetch ?? ((input, init) => fetch(input, init))
  const newXhr: () => XhrLike = options.xhr ?? (() => new XMLHttpRequest() as unknown as XhrLike)
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

  function sendXhr(path: string, form: FormData, accessToken: string | null, o: UploadOptions): Promise<Response> {
    return new Promise((resolve, reject) => {
      if (o.signal?.aborted) return reject(abortError())
      const xhr = newXhr()
      xhr.open('POST', baseUrl + path)
      xhr.withCredentials = true
      xhr.setRequestHeader('Accept', 'application/json')
      if (accessToken) xhr.setRequestHeader('Authorization', `Bearer ${accessToken}`)
      // No Content-Type: the browser must set multipart/form-data with its boundary itself.
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) o.onProgress?.(e.loaded, e.total)
      }
      xhr.onload = () => resolve(xhrToResponse(xhr))
      xhr.onerror = () => reject(new NetworkError())
      xhr.ontimeout = () => reject(new NetworkError('The upload timed out'))
      xhr.onabort = () => reject(abortError())
      o.signal?.addEventListener('abort', () => xhr.abort(), { once: true })
      xhr.send(form)
    })
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

  /** Sends with the current token; on 401 refreshes once and retries once; throws ApiError for any non-2xx. */
  async function withAuthRetry(path: string, sender: (accessToken: string | null) => Promise<Response>): Promise<Response> {
    const sentWith = token
    let res = await sender(sentWith)

    if (res.status === 401 && !AUTH_PATHS.includes(path)) {
      // If another request refreshed while ours was in flight, just retry with the newer token.
      const fresh = token !== null && token !== sentWith ? token : await refresh()
      if (!fresh) {
        expireSession()
        throw new ApiError(401, 'Your session has expired. Please log in again.')
      }
      res = await sender(fresh)
      if (res.status === 401) {
        expireSession()
        throw await toApiError(res)
      }
    }

    if (!res.ok) throw await toApiError(res)
    return res
  }

  async function parse<T>(res: Response): Promise<T> {
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
    async request<T>(path: string, o: RequestOptions = {}) {
      return parse<T>(await withAuthRetry(path, (t) => send(path, o, t)))
    },
    async upload<T>(path: string, form: FormData, o: UploadOptions = {}) {
      return parse<T>(await withAuthRetry(path, (t) => sendXhr(path, form, t, o)))
    },
  }
}
