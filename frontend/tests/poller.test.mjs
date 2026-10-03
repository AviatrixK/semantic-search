import test from 'node:test'
import assert from 'node:assert/strict'
import { nextDelay, startPolling } from '../.test-build/lib/poller.js'
import { ApiError } from '../.test-build/api/errors.js'

const tick = () => new Promise((r) => setTimeout(r, 0))
const until = async (cond, ms = 1000) => {
  const end = Date.now() + ms
  while (!cond()) { if (Date.now() > end) throw new Error('timed out waiting'); await tick() }
}

/** A sleep that records requested delays and resolves immediately (or on abort). */
function fakeSleep() {
  const delays = []
  return { delays, sleep: async (ms) => { delays.push(ms); await tick() } }
}

test('polls until isDone, one update per poll, sleeping the interval between polls', async () => {
  const { delays, sleep } = fakeSleep()
  const stages = ['queued', 'extracting', 'transcribing', 'done']
  let i = 0
  const updates = []
  startPolling({
    fetch: async () => ({ stage: stages[i++] }),
    isDone: (j) => j.stage === 'done',
    onUpdate: (j) => updates.push(j.stage),
    onError: () => assert.fail('no errors expected'),
    sleep,
  })
  await until(() => updates.length === 4)
  await tick(); await tick()
  assert.deepEqual(updates, stages)
  assert.deepEqual(delays, [2000, 2000, 2000]) // default 2 s, and no sleep after the final poll
  assert.equal(i, 4) // never polls again after done
})

test('stops on a failed job too (isDone decides)', async () => {
  const { sleep } = fakeSleep()
  let calls = 0
  startPolling({
    fetch: async () => { calls++; return { stage: calls < 2 ? 'queued' : 'failed' } },
    isDone: (j) => j.stage === 'done' || j.stage === 'failed',
    onUpdate: () => {}, onError: () => {}, sleep,
  })
  await until(() => calls === 2)
  await tick(); await tick()
  assert.equal(calls, 2)
})

test('stop() ends polling: no further requests and no late updates', async () => {
  let calls = 0
  const updates = []
  const stop = startPolling({
    fetch: async () => { calls++; return { n: calls } },
    isDone: () => false,
    onUpdate: (v) => updates.push(v.n),
    onError: () => {},
    intervalMs: 5,
  })
  await until(() => calls >= 3)
  stop()
  const at = calls
  await new Promise((r) => setTimeout(r, 40))
  assert.equal(calls, at)
})

test('stop() while a request is in flight: its result is discarded', async () => {
  let release
  const updates = []
  const stop = startPolling({
    fetch: () => new Promise((resolve) => { release = () => resolve({ stage: 'done' }) }),
    isDone: () => true,
    onUpdate: (v) => updates.push(v),
    onError: () => assert.fail('no error expected'),
  })
  await until(() => release)
  stop()
  release()
  await new Promise((r) => setTimeout(r, 10))
  assert.deepEqual(updates, [])
})

test('requests never overlap even when the server is slower than the interval', async () => {
  let active = 0, maxActive = 0, calls = 0
  startPolling({
    fetch: async () => {
      active++; maxActive = Math.max(maxActive, active); calls++
      await new Promise((r) => setTimeout(r, 15)) // slower than intervalMs
      active--
      return { done: calls >= 4 }
    },
    isDone: (v) => v.done,
    onUpdate: () => {}, onError: () => {},
    intervalMs: 2,
  })
  await until(() => calls >= 4, 2000)
  await new Promise((r) => setTimeout(r, 30))
  assert.equal(maxActive, 1)
})

test('transient errors are retried with growing delays; a success resets the counter', async () => {
  const { delays, sleep } = fakeSleep()
  const script = ['err', 'err', 'ok', 'err', 'done']
  let i = 0
  const errors = []
  const updates = []
  startPolling({
    fetch: async () => {
      const step = script[i++]
      if (step === 'err') throw new TypeError('network down')
      return { stage: step === 'done' ? 'done' : 'queued' }
    },
    isDone: (j) => j.stage === 'done',
    onUpdate: (j) => updates.push(j.stage),
    onError: (_e, info) => errors.push(info),
    sleep,
  })
  await until(() => updates.length === 2)
  assert.deepEqual(errors, [{ consecutive: 1, fatal: false }, { consecutive: 2, fatal: false }, { consecutive: 1, fatal: false }])
  assert.deepEqual(delays, [4000, 8000, 2000, 4000]) // 2s*2, 2s*4, success -> 2s, error again -> 2s*2
})

test('a fatal error (e.g. 404) stops polling immediately', async () => {
  const { sleep, delays } = fakeSleep()
  let calls = 0
  const errors = []
  startPolling({
    fetch: async () => { calls++; throw new ApiError(404, 'Not found') },
    isDone: () => false,
    onUpdate: () => assert.fail('no update expected'),
    onError: (e, info) => errors.push([e.status, info.fatal]),
    isFatal: (e) => e instanceof ApiError && e.status === 404,
    sleep,
  })
  await until(() => errors.length === 1)
  await tick(); await tick()
  assert.equal(calls, 1)
  assert.deepEqual(errors, [[404, true]])
  assert.deepEqual(delays, [])
})

test('429 waits for the server-provided Retry-After', async () => {
  const { sleep, delays } = fakeSleep()
  let calls = 0
  startPolling({
    fetch: async () => { calls++; if (calls === 1) throw new ApiError(429, 'slow down', 7); return { done: true } },
    isDone: (v) => v.done,
    onUpdate: () => {}, onError: () => {}, sleep,
  })
  await until(() => calls === 2)
  assert.deepEqual(delays, [7000])
})

test('nextDelay: interval, exponential backoff with a cap, and Retry-After never shorter than the interval', () => {
  assert.equal(nextDelay(2000, 10000, 0), 2000)
  assert.equal(nextDelay(2000, 10000, 1), 4000)
  assert.equal(nextDelay(2000, 10000, 2), 8000)
  assert.equal(nextDelay(2000, 10000, 3), 10000)
  assert.equal(nextDelay(2000, 10000, 9), 10000)
  assert.equal(nextDelay(2000, 10000, 1, new ApiError(429, 'x', 30)), 30000)
  assert.equal(nextDelay(2000, 10000, 1, new ApiError(429, 'x', 1)), 2000)
  assert.equal(nextDelay(2000, 10000, 1, new ApiError(500, 'x')), 4000)
})
