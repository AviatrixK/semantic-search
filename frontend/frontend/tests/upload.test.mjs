import test from 'node:test'
import assert from 'node:assert/strict'
import { createApiClient } from '../.test-build/api/client.js'
import { ApiError, NetworkError, describeError } from '../.test-build/api/errors.js'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

/** Fake XMLHttpRequest. `behave(xhr, n)` plays the server: report progress, then set status/body and call xhr.onload(). */
class FakeXhr {
  constructor(behave, n, log) {
    this.behave = behave; this.n = n; this.log = log
    this.headers = {}; this.upload = { onprogress: null }; this.status = 0; this.responseText = ''; this.responseHeaders = ''
    this.withCredentials = false
  }
  open(method, url) { this.method = method; this.url = url }
  setRequestHeader(name, value) { this.headers[name] = value }
  getAllResponseHeaders() { return this.responseHeaders }
  send(form) { this.form = form; setTimeout(() => this.behave(this, this.n), 0) }
  abort() { this.aborted = true; this.onabort?.() }
  respond(status, body, headers = {}) {
    this.status = status
    this.responseText = body === undefined ? '' : JSON.stringify(body)
    this.responseHeaders = Object.entries({ 'Content-Type': 'application/json', ...headers }).map(([k, v]) => `${k}: ${v}`).join('\r\n')
    this.onload()
  }
}

function setup(behave, fetchHandler = () => json({ access_token: 'new' })) {
  const instances = []
  const fetchCalls = []
  const client = createApiClient({
    fetch: async (url, init) => { fetchCalls.push(url); return fetchHandler(url, init) },
    xhr: () => { const x = new FakeXhr(behave, instances.length + 1); instances.push(x); return x },
  })
  const expired = []
  client.onSessionExpired(() => expired.push(true))
  return { client, instances, fetchCalls, expired }
}

const form = () => { const f = new FormData(); f.append('file', new Blob(['x']), 'a.mp4'); return f }

test('upload: POSTs the form with auth and credentials, reports progress, returns parsed JSON', async () => {
  const { client, instances } = setup((xhr) => {
    xhr.upload.onprogress({ lengthComputable: true, loaded: 25, total: 100 })
    xhr.upload.onprogress({ lengthComputable: false, loaded: 0, total: 0 }) // ignored
    xhr.upload.onprogress({ lengthComputable: true, loaded: 100, total: 100 })
    xhr.respond(202, { video_id: 'v1', job_id: 'j1' })
  })
  client.setToken('tok')
  const seen = []
  const res = await client.upload('/api/videos', form(), { onProgress: (l, t) => seen.push([l, t]) })
  assert.deepEqual(res, { video_id: 'v1', job_id: 'j1' })
  assert.deepEqual(seen, [[25, 100], [100, 100]])
  const x = instances[0]
  assert.equal(x.method, 'POST')
  assert.equal(x.url, '/api/videos')
  assert.equal(x.withCredentials, true)
  assert.equal(x.headers.Authorization, 'Bearer tok')
  assert.equal('Content-Type' in x.headers, false) // the browser must add the multipart boundary itself
  assert.ok(x.form instanceof FormData)
})

test('upload: a 401 refreshes once and retries the whole upload with the new token', async () => {
  const { client, instances, fetchCalls } = setup((xhr, n) => {
    xhr.upload.onprogress({ lengthComputable: true, loaded: 10 * n, total: 100 })
    if (xhr.headers.Authorization === 'Bearer new') xhr.respond(202, { video_id: 'v', job_id: 'j' })
    else xhr.respond(401, { detail: 'Invalid or expired token' })
  })
  client.setToken('old')
  const seen = []
  const res = await client.upload('/api/videos', form(), { onProgress: (l) => seen.push(l) })
  assert.equal(res.job_id, 'j')
  assert.equal(instances.length, 2)
  assert.deepEqual(fetchCalls, ['/auth/refresh'])
  assert.deepEqual(seen, [10, 20]) // progress restarts on the retry
})

test('upload: parallel uploads that all hit 401 share ONE refresh', async () => {
  const { client, fetchCalls } = setup(
    (xhr) => (xhr.headers.Authorization === 'Bearer new' ? xhr.respond(202, { video_id: 'v', job_id: 'j' }) : xhr.respond(401, { detail: 'expired' })),
    async () => { await sleep(10); return json({ access_token: 'new' }) },
  )
  client.setToken('old')
  await Promise.all([1, 2, 3].map(() => client.upload('/api/videos', form())))
  assert.equal(fetchCalls.length, 1)
})

test('upload: refresh rejected -> session expires, error thrown', async () => {
  const { client, expired } = setup((xhr) => xhr.respond(401, { detail: 'expired' }), () => json({ detail: 'invalid' }, 401))
  client.setToken('old')
  await assert.rejects(client.upload('/api/videos', form()), (e) => e instanceof ApiError && e.status === 401)
  assert.equal(expired.length, 1)
})

test('upload: 413 becomes a friendly message with the limit', async () => {
  const { client } = setup((xhr) => xhr.respond(413, { detail: 'File too large (max 500 MB)' }))
  await assert.rejects(client.upload('/api/videos', form()), (e) => {
    assert.equal(e.status, 413)
    assert.equal(describeError(e, 'upload'), 'This file is too large. The limit is 500 MB.')
    return true
  })
})

test('upload: 415 and 400 and 403 are friendly', async () => {
  for (const [status, detail, expected] of [
    [415, 'Unsupported file extension. Allowed: mkv, mov, mp4, webm', 'Unsupported file type. Upload an MP4, WebM, MOV or MKV video.'],
    [415, 'Unsupported type text/plain', 'Unsupported file type. Upload an MP4, WebM, MOV or MKV video.'],
    [400, 'Empty file', 'That file is empty.'],
    [403, 'Admins only', 'Only administrators can upload videos.'],
  ]) {
    const { client } = setup((xhr) => xhr.respond(status, { detail }))
    await assert.rejects(client.upload('/api/videos', form()), (e) => describeError(e, 'upload') === expected)
  }
})

test('upload: 429 uses Retry-After in the message', async () => {
  const { client } = setup((xhr) => xhr.respond(429, { detail: 'Too many requests' }, { 'Retry-After': '30' }))
  await assert.rejects(client.upload('/api/videos', form()), (e) => {
    assert.equal(e.retryAfter, 30)
    assert.equal(describeError(e, 'upload'), 'Too many attempts. Please wait 30 seconds and try again.')
    return true
  })
})

test('upload: the readable 500 from a failed database write is not shown raw', async () => {
  const { client } = setup((xhr) => xhr.respond(500, { detail: 'Could not save the video. Please try again.' }))
  await assert.rejects(client.upload('/api/videos', form()), (e) => /went wrong on the server/.test(describeError(e, 'upload')))
})

test('upload: network failure and timeout become NetworkError', async () => {
  const a = setup((xhr) => xhr.onerror())
  await assert.rejects(a.client.upload('/api/videos', form()), (e) => e instanceof NetworkError)
  const b = setup((xhr) => xhr.ontimeout())
  await assert.rejects(b.client.upload('/api/videos', form()), (e) => e instanceof NetworkError)
})

test('upload: aborting via the signal aborts the XHR and rejects with AbortError', async () => {
  const { client, instances } = setup(() => { /* never answers */ })
  const ctrl = new AbortController()
  const p = client.upload('/api/videos', form(), { signal: ctrl.signal })
  await sleep(5)
  ctrl.abort()
  await assert.rejects(p, (e) => e.name === 'AbortError')
  assert.equal(instances[0].aborted, true)
})

test('upload: an already-aborted signal never sends anything', async () => {
  const { client, instances } = setup(() => {})
  const ctrl = new AbortController()
  ctrl.abort()
  await assert.rejects(client.upload('/api/videos', form(), { signal: ctrl.signal }), (e) => e.name === 'AbortError')
  assert.equal(instances.length, 0)
})

test('upload: 204 with no body resolves to undefined', async () => {
  const { client } = setup((xhr) => xhr.respond(204, undefined, {}))
  assert.equal(await client.upload('/x', form()), undefined)
})

test('describeError: reprocess, delete and job contexts', () => {
  assert.equal(describeError(new ApiError(409, 'This video is already being processed'), 'reprocess'), 'This video is already being processed.')
  assert.equal(describeError(new ApiError(404, 'Not found'), 'reprocess'), 'That video no longer exists.')
  assert.equal(describeError(new ApiError(404, 'Not found'), 'delete'), 'That video no longer exists.')
  assert.match(describeError(new ApiError(404, 'Not found'), 'job'), /no longer exists/)
})
