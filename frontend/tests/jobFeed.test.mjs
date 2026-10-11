import test from 'node:test'
import assert from 'node:assert/strict'
import { connectJobFeed, endedWhileAway, hasActiveJob, jobsReducer, parseJobEvent, reconnectDelay } from '../.test-build/lib/jobFeed.js'
import { ApiError } from '../.test-build/api/errors.js'

const job = (id, stage, progress = 0, extra = {}) => ({ id, video_id: 'v-' + id, stage, progress, error: null, ...extra })
const sse = (event, data) => ({ event, data: JSON.stringify(data) })
const tick = () => new Promise((r) => setTimeout(r, 0))
const until = async (cond, ms = 1000) => {
  const end = Date.now() + ms
  while (!cond()) { if (Date.now() > end) throw new Error('timed out waiting'); await tick() }
}

// ---- parsing
test('parses a snapshot and a job event', () => {
  assert.deepEqual(parseJobEvent(sse('snapshot', { jobs: [job('a', 'queued')] })), { type: 'snapshot', jobs: [job('a', 'queued')] })
  assert.deepEqual(parseJobEvent(sse('job', job('a', 'embedding', 70))), { type: 'job', job: job('a', 'embedding', 70) })
})

test('malformed or unknown events are ignored, never thrown', () => {
  assert.equal(parseJobEvent({ event: 'job', data: 'not json' }), null)
  assert.equal(parseJobEvent(sse('job', { id: 1 })), null)
  assert.equal(parseJobEvent(sse('snapshot', { jobs: 'nope' })), null)
  assert.equal(parseJobEvent(sse('other', {})), null)
  assert.equal(parseJobEvent(sse('snapshot', { jobs: [job('a', 'queued'), { bad: true }] })).jobs.length, 1)
})

test('progress is clamped to 0..100', () => {
  assert.equal(parseJobEvent(sse('job', job('a', 'embedding', 250))).job.progress, 100)
  assert.equal(parseJobEvent(sse('job', job('a', 'embedding', -4))).job.progress, 0)
})

// ---- reducer
test('snapshot replaces running jobs and keeps finished ones', () => {
  let s = jobsReducer({}, { type: 'snapshot', jobs: [job('a', 'transcribing', 40), job('b', 'queued')] })
  s = jobsReducer(s, { type: 'job', job: job('a', 'done', 100) })
  s = jobsReducer(s, { type: 'snapshot', jobs: [job('c', 'queued')] }) // b ended while away, c is new
  assert.deepEqual(Object.keys(s).sort(), ['a', 'c'])
  assert.equal(s.a.stage, 'done')
})

test('a late queued event can never undo done', () => {
  const s = jobsReducer({}, { type: 'job', job: job('a', 'done', 100) })
  assert.equal(jobsReducer(s, { type: 'job', job: job('a', 'queued') }), s)
})

test('dismiss removes one job; unknown ids change nothing', () => {
  const s = jobsReducer({}, { type: 'snapshot', jobs: [job('a', 'queued'), job('b', 'queued')] })
  assert.deepEqual(Object.keys(jobsReducer(s, { type: 'dismiss', id: 'a' })), ['b'])
  assert.equal(jobsReducer(s, { type: 'dismiss', id: 'zzz' }), s)
})

test('hasActiveJob: true only while a job is running', () => {
  assert.equal(hasActiveJob({}), false)
  assert.equal(hasActiveJob({ a: job('a', 'embedding') }), true)
  assert.equal(hasActiveJob({ a: job('a', 'done', 100), b: job('b', 'failed') }), false)
})

test('endedWhileAway spots running jobs missing from a new snapshot', () => {
  assert.equal(endedWhileAway({ a: job('a', 'embedding') }, [job('a', 'embedding', 80)]), false)
  assert.equal(endedWhileAway({ a: job('a', 'embedding') }, []), true)
  assert.equal(endedWhileAway({ a: job('a', 'done', 100) }, []), false)
})

test('reconnect delay doubles and is capped', () => {
  assert.deepEqual([1, 2, 3, 4, 10].map((n) => reconnectDelay(n)), [1000, 2000, 4000, 8000, 30000])
})

// ---- connection
function harness(streamImpl) {
  const delays = []
  const statuses = []
  const events = []
  const calls = []
  const stop = connectJobFeed({
    stream: (path, o) => { calls.push(path); return streamImpl(calls.length, o) },
    onEvent: (e) => events.push(e),
    onStatus: (s) => { if (statuses.at(-1) !== s) statuses.push(s) },
    sleep: async (ms) => { delays.push(ms); await tick() },
  })
  return { stop, delays, statuses, events, calls }
}

test('opens ONE connection to /api/jobs/events and delivers events while it stays healthy', async () => {
  let release
  const h = harness((n, o) => new Promise((resolve) => {
    o.onEvent(sse('snapshot', { jobs: [job('a', 'queued')] }))
    o.onEvent(sse('job', job('a', 'transcribing', 40)))
    release = resolve
  }))
  await until(() => h.events.length === 2)
  await tick(); await tick()
  assert.deepEqual(h.calls, ['/api/jobs/events']) // no polling: still the first and only request
  assert.deepEqual(h.statuses, ['connecting', 'live'])
  assert.deepEqual(h.events.map((e) => e.type), ['snapshot', 'job'])
  h.stop()
  release()
})

test('reconnects with growing pauses after a drop and starts over after a snapshot', async () => {
  const h = harness(async (n, o) => {
    if (n <= 2) throw new Error('network') // two failures in a row
    if (n === 3) { o.onEvent(sse('snapshot', { jobs: [] })); throw new Error('dropped after working') }
    return new Promise(() => {}) // the 4th attempt stays open
  })
  await until(() => h.calls.length === 4)
  assert.deepEqual(h.delays, [1000, 2000, 1000]) // the successful snapshot reset the backoff
  h.stop()
})

test('a server that ends the stream is reconnected too', async () => {
  const h = harness(async (n) => { if (n === 1) return; return new Promise(() => {}) })
  await until(() => h.calls.length === 2)
  assert.deepEqual(h.delays, [1000])
  h.stop()
})

test('401/403/404 stop for good (not an admin, or the session ended)', async () => {
  for (const status of [401, 403, 404]) {
    const h = harness(async () => { throw new ApiError(status, 'nope') })
    await until(() => h.statuses.includes('stopped'))
    await tick(); await tick()
    assert.equal(h.calls.length, 1)
    assert.deepEqual(h.delays, [])
  }
})

test('503 is retried with backoff', async () => {
  const h = harness(async (n) => { if (n === 1) throw new ApiError(503, 'unavailable'); return new Promise(() => {}) })
  await until(() => h.calls.length === 2)
  h.stop()
})

test('stop() closes the connection and nothing reconnects afterwards', async () => {
  let signal
  const h = harness((n, o) => { signal = o.signal; return new Promise(() => {}) })
  await until(() => h.calls.length === 1)
  h.stop()
  assert.equal(signal.aborted, true)
  await tick(); await tick()
  assert.equal(h.calls.length, 1)
})
