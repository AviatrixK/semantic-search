import test from 'node:test'
import assert from 'node:assert/strict'
import { createApiClient } from '../.test-build/api/client.js'
import { ApiError, NetworkError } from '../.test-build/api/errors.js'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const json = (body, status = 200, headers = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...headers } })

/** fetch stand-in that records every call. handler(call, n) returns a Response (or throws). */
function recorder(handler) {
  const calls = []
  const fetch = async (url, init = {}) => {
    const call = { url, method: init.method, auth: init.headers?.Authorization, init }
    calls.push(call)
    return handler(call, calls.length)
  }
  return { fetch, calls }
}

function newClient(handler, extra = {}) {
  const { fetch, calls } = recorder(handler)
  const client = createApiClient({ fetch, ...extra })
  const expired = []
  client.onSessionExpired(() => expired.push(true))
  return { client, calls, expired }
}

/** A typical backend: /api/data needs "Bearer new"; /auth/refresh hands out "new". */
const refreshingBackend = async (call) => {
  if (call.url === '/auth/refresh') {
    await sleep(10)
    return json({ access_token: 'new', token_type: 'bearer' })
  }
  return call.auth === 'Bearer new' ? json({ ok: true }) : json({ detail: 'Invalid or expired token' }, 401)
}

test('adds the bearer token, sends credentials, and serialises json bodies', async () => {
  const { client, calls } = newClient(() => json({ ok: true }))
  client.setToken('tok')
  await client.request('/api/x')
  await client.request('/api/y', { method: 'POST', json: { a: 1 } })
  assert.equal(calls[0].auth, 'Bearer tok')
  assert.equal(calls[0].init.credentials, 'include')
  assert.equal(calls[0].method, 'GET')
  assert.equal(calls[1].init.body, '{"a":1}')
  assert.equal(calls[1].init.headers['Content-Type'], 'application/json')
})

test('no Authorization header before login', async () => {
  const { client, calls } = newClient(() => json({}))
  await client.request('/api/x')
  assert.equal(calls[0].auth, undefined)
})

test('401 triggers one refresh, then the request is retried with the new token', async () => {
  const { client, calls } = newClient(refreshingBackend)
  client.setToken('old')
  assert.deepEqual(await client.request('/api/data'), { ok: true })
  assert.deepEqual(calls.map((c) => [c.method, c.url, c.auth]), [
    ['GET', '/api/data', 'Bearer old'],
    ['POST', '/auth/refresh', undefined],
    ['GET', '/api/data', 'Bearer new'],
  ])
  assert.equal(client.getToken(), 'new')
})

test('many parallel 401s share ONE refresh request', async () => {
  const { client, calls } = newClient(refreshingBackend)
  client.setToken('old')
  const results = await Promise.all(Array.from({ length: 6 }, () => client.request('/api/data')))
  assert.equal(results.length, 6)
  assert.equal(calls.filter((c) => c.url === '/auth/refresh').length, 1)
})

test('two concurrent refresh() calls (StrictMode double effect, two components) send one request', async () => {
  const { client, calls } = newClient(refreshingBackend)
  const [a, b] = await Promise.all([client.refresh(), client.refresh()])
  assert.equal(a, 'new')
  assert.equal(b, 'new')
  assert.equal(calls.length, 1)
  await client.refresh() // the in-flight slot was released, so a later refresh is a new request
  assert.equal(calls.length, 2)
})

test('refresh rejected (401): logs out, rejects, and does not retry or loop', async () => {
  const { client, calls, expired } = newClient((call) =>
    call.url === '/auth/refresh' ? json({ detail: 'Refresh token invalid' }, 401) : json({ detail: 'nope' }, 401))
  client.setToken('old')
  await assert.rejects(client.request('/api/data'), (e) => e instanceof ApiError && e.status === 401)
  assert.equal(calls.length, 2) // original + one refresh, nothing more
  assert.equal(client.getToken(), null)
  assert.equal(expired.length, 1)
})

test('still 401 after a successful refresh: logs out instead of refreshing again', async () => {
  const { client, calls, expired } = newClient((call) =>
    call.url === '/auth/refresh' ? json({ access_token: 'new' }) : json({ detail: 'nope' }, 401))
  client.setToken('old')
  await assert.rejects(client.request('/api/data'), (e) => e.status === 401)
  assert.equal(calls.filter((c) => c.url === '/auth/refresh').length, 1)
  assert.equal(calls.length, 3)
  assert.equal(expired.length, 1)
})

test('a request that lost the race reuses the token another request already refreshed', async () => {
  const { client, calls } = newClient(async (call) => {
    if (call.url === '/auth/refresh') throw new Error('must not be called')
    if (call.auth === 'Bearer old') {
      await sleep(20)
      return json({ detail: 'expired' }, 401)
    }
    return json({ ok: true })
  })
  client.setToken('old')
  const pending = client.request('/api/data')
  await sleep(5)
  client.setToken('newer') // another request finished refreshing while ours was in flight
  assert.deepEqual(await pending, { ok: true })
  assert.deepEqual(calls.map((c) => c.auth), ['Bearer old', 'Bearer newer'])
})

test('backend unreachable during refresh: surfaces a NetworkError and keeps the session', async () => {
  const { client, expired } = newClient((call) => {
    if (call.url === '/auth/refresh') throw new TypeError('fetch failed')
    return json({ detail: 'expired' }, 401)
  })
  client.setToken('old')
  await assert.rejects(client.request('/api/data'), (e) => e instanceof NetworkError)
  assert.equal(expired.length, 0)
  assert.equal(client.getToken(), 'old')
})

test('refresh answering 5xx does not log the user out', async () => {
  const { client, expired } = newClient((call) =>
    call.url === '/auth/refresh' ? json({ detail: 'db down' }, 503) : json({ detail: 'expired' }, 401))
  client.setToken('old')
  await assert.rejects(client.request('/api/data'), (e) => e instanceof ApiError && e.status === 503)
  assert.equal(expired.length, 0)
})

test('auth endpoints never trigger a refresh (wrong password is just a 401)', async () => {
  const { client, calls, expired } = newClient(() => json({ detail: 'Invalid credentials' }, 401))
  await assert.rejects(client.request('/auth/login', { method: 'POST', json: { email: 'a@b.c', password: 'x' } }),
    (e) => e.status === 401)
  assert.equal(calls.length, 1)
  assert.equal(expired.length, 0)
})

test('429 carries Retry-After seconds and the server message', async () => {
  const { client } = newClient(() =>
    json({ detail: 'Too many requests. Try again in 42 seconds.' }, 429, { 'Retry-After': '42' }))
  await assert.rejects(client.request('/auth/login', { method: 'POST', json: {} }), (e) => {
    assert.ok(e instanceof ApiError)
    assert.equal(e.status, 429)
    assert.equal(e.retryAfter, 42)
    assert.match(e.message, /Too many requests/)
    return true
  })
})

test('429 without Retry-After leaves retryAfter undefined', async () => {
  const { client } = newClient(() => json({ detail: 'slow down' }, 429))
  await assert.rejects(client.request('/api/x'), (e) => e.status === 429 && e.retryAfter === undefined)
})

test('204 and non-JSON success return undefined; network failure becomes NetworkError', async () => {
  const noContent = newClient(() => new Response(null, { status: 204 }))
  assert.equal(await noContent.client.request('/auth/logout', { method: 'POST' }), undefined)
  const net = newClient(() => { throw new TypeError('fetch failed') })
  await assert.rejects(net.client.request('/api/x'), (e) => e instanceof NetworkError)
})

test('token listeners fire on change only', async () => {
  const { client } = newClient(() => json({}))
  const seen = []
  const off = client.onTokenChange((t) => seen.push(t))
  client.setToken('a')
  client.setToken('a')
  client.setToken(null)
  off()
  client.setToken('b')
  assert.deepEqual(seen, ['a', null])
})

// ---------------------------------------------------------------------------------------------------------------
// Two browser tabs. A simulation of the backend's refresh rules (app/api/routes/auth.py): refresh tokens are
// single-use, and presenting an already-used one revokes EVERY session. Both tabs share one cookie jar, and the
// browser attaches the cookie at the moment a request is sent.
function sharedBrowser() {
  let n = 1
  const valid = new Set(['R1'])
  const used = new Set()
  const state = { cookie: 'R1', revokedAll: false, rotations: 0 }
  async function fetchForTab(url) {
    assert.equal(url, '/auth/refresh')
    const sent = state.cookie // cookie read when the request leaves the tab
    await sleep(15) // network + server time
    if (!valid.has(sent)) {
      if (used.has(sent)) { valid.clear(); state.revokedAll = true } // reuse detection
      return json({ detail: 'Refresh token invalid' }, 401)
    }
    valid.delete(sent)
    used.add(sent)
    const next = `R${++n}`
    valid.add(next)
    state.rotations++
    state.cookie = next // Set-Cookie is applied before the page sees the response
    return json({ access_token: `T${n}` })
  }
  return { state, fetchForTab }
}

function mutex() {
  let tail = Promise.resolve()
  return (fn) => {
    const run = tail.then(fn, fn)
    tail = run.then(() => undefined, () => undefined)
    return run
  }
}

test('two tabs refreshing at the same moment WITHOUT a cross-tab lock trip reuse detection (the hazard)', async () => {
  const { state, fetchForTab } = sharedBrowser()
  const tabA = createApiClient({ fetch: fetchForTab })
  const tabB = createApiClient({ fetch: fetchForTab })
  const [a, b] = await Promise.all([tabA.refresh(), tabB.refresh()])
  assert.equal(state.revokedAll, true)
  assert.equal([a, b].filter((t) => t === null).length, 1)
})

test('two tabs refreshing at the same moment WITH the cross-tab lock never revoke anything', async () => {
  const { state, fetchForTab } = sharedBrowser()
  const lock = mutex() // stands in for navigator.locks: one lock shared by every tab
  const tabA = createApiClient({ fetch: fetchForTab, runExclusive: lock })
  const tabB = createApiClient({ fetch: fetchForTab, runExclusive: lock })
  const [a, b] = await Promise.all([tabA.refresh(), tabB.refresh()])
  assert.equal(state.revokedAll, false)
  assert.ok(a && b && a !== b)
  assert.equal(state.rotations, 2) // B waited, then rotated the cookie A had just stored
})

test('page load in 3 tabs, each with StrictMode double bootstrap: 6 callers, no revocation, 3 server refreshes', async () => {
  const { state, fetchForTab } = sharedBrowser()
  const lock = mutex()
  const tabs = [0, 1, 2].map(() => createApiClient({ fetch: fetchForTab, runExclusive: lock }))
  const results = await Promise.all(tabs.flatMap((t) => [t.refresh(), t.refresh()]))
  assert.equal(state.revokedAll, false)
  assert.ok(results.every((t) => t !== null))
  assert.equal(state.rotations, 3) // one per tab: the duplicate call inside each tab joined the in-flight one
})
