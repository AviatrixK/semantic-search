import test from 'node:test'
import assert from 'node:assert/strict'
import { createApiClient } from '../.test-build/api/client.js'
import { ApiError, NetworkError } from '../.test-build/api/errors.js'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const enc = new TextEncoder()
const json = (body, status = 200, headers = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...headers } })

/** A streaming Response whose chunks the test releases one at a time. */
function controlledStream() {
  let controller
  const body = new ReadableStream({ start(c) { controller = c } })
  return {
    response: new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
    push: (bytes) => controller.enqueue(typeof bytes === 'string' ? enc.encode(bytes) : bytes),
    close: () => controller.close(),
    fail: (err) => controller.error(err),
  }
}

function setup(handler) {
  const calls = []
  const client = createApiClient({ fetch: async (url, init = {}) => { const c = { url, init, auth: init.headers?.Authorization }; calls.push(c); return handler(c, calls.length) } })
  client.setToken('tok')
  return { client, calls }
}

const sse = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`

test('stream: sends the bearer token and Accept: text/event-stream, and resolves when the server ends the stream', async () => {
  const { client, calls } = setup(() => {
    const s = controlledStream()
    s.push(sse('job', { stage: 'queued' })); s.push(sse('done', {})); s.close()
    return s.response
  })
  const got = []
  await client.stream('/api/jobs/events', { onEvent: (e) => got.push(e.event) })
  assert.deepEqual(got, ['job', 'done'])
  assert.equal(calls[0].auth, 'Bearer tok')
  assert.equal(calls[0].init.headers.Accept, 'text/event-stream')
  assert.equal(calls[0].init.credentials, 'include')
})

test('stream: events reach the caller AS THEY ARRIVE, not when the stream ends', async () => {
  const s = controlledStream()
  const { client } = setup(() => s.response)
  const got = []
  const done = client.stream('/x', { onEvent: (e) => got.push(e.event) })
  s.push(sse('step_start', { step: 1 }))
  await sleep(20)
  assert.deepEqual(got, ['step_start']) // delivered while the stream is still open
  s.push(sse('step_result', { step: 1 }))
  await sleep(20)
  assert.deepEqual(got, ['step_start', 'step_result'])
  s.close()
  await done
})

test('stream: a multi-byte character split across two network chunks is decoded correctly', async () => {
  const bytes = enc.encode(sse('answer', { answer: 'café 🎙 ok' }))
  const cut = bytes.indexOf(0xc3) + 1 // between the two bytes of "é"
  const { client } = setup(() => {
    const s = controlledStream()
    s.push(bytes.slice(0, cut)); s.push(bytes.slice(cut)); s.close()
    return s.response
  })
  const got = []
  await client.stream('/x', { onEvent: (e) => got.push(JSON.parse(e.data)) })
  assert.deepEqual(got, [{ answer: 'café 🎙 ok' }])
  // and an emoji (4 bytes) cut into single bytes
  const emoji = enc.encode(sse('answer', { answer: '🎙️' }))
  const { client: c2 } = setup(() => {
    const s = controlledStream()
    for (const b of emoji) s.push(new Uint8Array([b]))
    s.close()
    return s.response
  })
  const got2 = []
  await c2.stream('/x', { onEvent: (e) => got2.push(JSON.parse(e.data).answer) })
  assert.deepEqual(got2, ['🎙️'])
})

test('stream: a 401 before the stream refreshes the token once and retries', async () => {
  const { client, calls } = setup((c) => {
    if (c.url === '/auth/refresh') return json({ access_token: 'new' })
    if (c.auth === 'Bearer new') { const s = controlledStream(); s.push(sse('done', {})); s.close(); return s.response }
    return json({ detail: 'expired' }, 401)
  })
  const got = []
  await client.stream('/x', { onEvent: (e) => got.push(e.event) })
  assert.deepEqual(calls.map((c) => [c.url, c.auth]), [['/x', 'Bearer tok'], ['/auth/refresh', undefined], ['/x', 'Bearer new']])
  assert.deepEqual(got, ['done'])
})

test('stream: errors before the stream are ordinary ApiErrors with Retry-After (rate limit, validation)', async () => {
  for (const [status, headers, check] of [
    [429, { 'Retry-After': '42' }, (e) => e.retryAfter === 42],
    [422, {}, (e) => e.message === 'Invalid request'],
    [503, {}, (e) => e.status === 503],
  ]) {
    const { client } = setup(() => json({ detail: status === 422 ? 'Invalid request' : 'nope' }, status, headers))
    let seen = false
    await assert.rejects(client.stream('/x', { onEvent: () => { seen = true } }), (e) => e instanceof ApiError && e.status === status && check(e))
    assert.equal(seen, false)
  }
})

test('stream: aborting mid-stream rejects with AbortError', async () => {
  const ctrl = new AbortController()
  const { client } = setup((c) => {
    const s = controlledStream()
    c.init.signal.addEventListener('abort', () => s.fail(new DOMException('Aborted', 'AbortError')))
    s.push(sse('job', { stage: 'embedding' }))
    return s.response
  })
  const p = client.stream('/x', { signal: ctrl.signal, onEvent: () => {} })
  await sleep(20)
  ctrl.abort()
  await assert.rejects(p, (e) => e.name === 'AbortError')
})

test('stream: a connection that drops mid-stream becomes a NetworkError, after delivering what arrived', async () => {
  const s = controlledStream()
  const { client } = setup(() => s.response)
  const got = []
  const p = client.stream('/x', { onEvent: (e) => got.push(e.event) })
  s.push(sse('step_start', { step: 1 }))
  await sleep(20)
  s.fail(new TypeError('network error'))
  await assert.rejects(p, (e) => e instanceof NetworkError)
  assert.deepEqual(got, ['step_start'])
})

test('stream: a response without a body (very old browsers) is a clear NetworkError', async () => {
  const { client } = setup(() => ({ ok: true, status: 200, headers: new Headers(), body: null }))
  await assert.rejects(client.stream('/x', { onEvent: () => {} }), (e) => e instanceof NetworkError && /cannot read streamed/.test(e.message))
})
