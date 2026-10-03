import test from 'node:test'
import assert from 'node:assert/strict'
import { createSearcher } from '../.test-build/lib/searcher.js'
import { createStreamUrlCache } from '../.test-build/lib/streamUrls.js'
import { excerpt } from '../.test-build/lib/highlight.js'
import { moveActive, scrollTargetFor } from '../.test-build/lib/listNav.js'
import { ApiError } from '../.test-build/api/errors.js'

const tick = () => new Promise((r) => setTimeout(r, 0))
const hit = (n) => ({ video_id: 'v', title: 'T', start_sec: n, end_sec: n + 30, text: `t${n}`, score: 0.5, highlight: null })

/** Manual clock for debounce: schedule() queues, flush() fires whatever has not been cancelled. */
function manualScheduler() {
  const tasks = []
  return {
    delays: [],
    schedule(fn, ms) { const t = { fn, cancelled: false }; tasks.push(t); this.delays.push(ms); return () => { t.cancelled = true } },
    flush() { const due = tasks.splice(0).filter((t) => !t.cancelled); due.forEach((t) => t.fn()); return due.length },
    pending: () => tasks.filter((t) => !t.cancelled).length,
  }
}

function makeSearcher(runImpl) {
  const sched = manualScheduler()
  const states = []
  const calls = []
  const searcher = createSearcher({
    run: (q, signal) => { calls.push({ q, signal }); return runImpl(q, signal, calls.length) },
    onState: (s) => states.push(s),
    describe: (e) => ({ message: e.message, retryAfter: e instanceof ApiError ? e.retryAfter : undefined }),
    schedule: (fn, ms) => sched.schedule(fn, ms),
  })
  return { searcher, sched, states, calls, last: () => states.at(-1) }
}

test('searcher: waits 400 ms of silence, then searches once for the final text', async () => {
  const { searcher, sched, calls, last } = makeSearcher(async (q) => [hit(q.length)])
  for (const q of ['pu', 'pub', 'publ', 'public']) searcher.setQuery(q)
  assert.equal(calls.length, 0)
  assert.equal(sched.pending(), 1) // every keystroke replaced the previous timer
  assert.equal(sched.delays[0], 400)
  sched.flush()
  await tick()
  assert.deepEqual(calls.map((c) => c.q), ['public'])
  assert.equal(last().status, 'ready')
  assert.equal(last().query, 'public')
  assert.equal(last().hits.length, 1)
})

test('searcher: Enter searches immediately and cancels the pending debounce', async () => {
  const { searcher, sched, calls, last } = makeSearcher(async () => [hit(1)])
  searcher.setQuery('speaking')
  searcher.submit()
  await tick()
  assert.equal(calls.length, 1)
  assert.equal(sched.flush(), 0) // the timer was cancelled: no second request
  assert.equal(last().status, 'ready')
})

test('searcher: queries shorter than 2 characters never hit the server and clear the results', async () => {
  const { searcher, sched, calls, last } = makeSearcher(async () => [hit(1)])
  searcher.setQuery('hello'); sched.flush(); await tick()
  assert.equal(last().hits.length, 1)
  searcher.setQuery('h')
  assert.equal(last().status, 'idle')
  assert.deepEqual(last().hits, [])
  assert.equal(sched.flush(), 0)
  searcher.setQuery('   ')
  searcher.submit()
  assert.equal(calls.length, 1)
})

test('searcher: a new query aborts the one in flight and only the newest answer is shown', async () => {
  const resolvers = []
  const { searcher, sched, calls, last } = makeSearcher((q) => new Promise((resolve) => resolvers.push(() => resolve([hit(q.length)]))))
  searcher.setQuery('first'); sched.flush()
  searcher.setQuery('second query'); sched.flush()
  assert.equal(calls[0].signal.aborted, true)
  assert.equal(calls[1].signal.aborted, false)
  resolvers[1](); await tick()
  resolvers[0](); await tick() // the slow, outdated response arrives last
  assert.equal(last().query, 'second query')
  assert.equal(last().hits[0].start_sec, 'second query'.length)
})

test('searcher: an unchanged query is not searched again', async () => {
  const { searcher, sched, calls } = makeSearcher(async () => [hit(1)])
  searcher.setQuery('same'); sched.flush(); await tick()
  searcher.setQuery('same'); searcher.submit(); await tick()
  assert.equal(calls.length, 1)
  searcher.setQuery('same '); // whitespace-only change
  assert.equal(sched.pending(), 0)
})

test('searcher: typing away and back to the displayed query cancels the extra search', async () => {
  const { searcher, sched, calls, last } = makeSearcher(async () => [hit(1)])
  searcher.setQuery('robots'); sched.flush(); await tick()
  searcher.setQuery('robots teach')
  searcher.setQuery('robots')
  assert.equal(sched.pending(), 0)
  assert.equal(calls.length, 1)
  assert.equal(last().status, 'ready')
})

test('searcher: previous results stay visible while the next search loads', async () => {
  let release
  const { searcher, sched, states, last } = makeSearcher((q) => (q === 'one one' ? Promise.resolve([hit(1)]) : new Promise((r) => { release = () => r([hit(2)]) })))
  searcher.setQuery('one one'); sched.flush(); await tick()
  searcher.setQuery('two two'); sched.flush()
  assert.equal(last().status, 'loading')
  assert.equal(last().hits[0].start_sec, 1)
  release(); await tick()
  assert.equal(last().hits[0].start_sec, 2)
  assert.ok(states.length >= 4)
})

test('searcher: errors are reported with Retry-After for 429, and Enter retries the same query', async () => {
  let n = 0
  const { searcher, sched, calls, last } = makeSearcher(async () => {
    if (++n === 1) throw new ApiError(429, 'Too many requests', 17)
    return [hit(9)]
  })
  searcher.setQuery('rate limited'); sched.flush(); await tick()
  assert.equal(last().status, 'error')
  assert.equal(last().retryAfter, 17)
  assert.match(last().error, /Too many requests/)
  searcher.submit(); await tick()
  assert.equal(calls.length, 2)
  assert.equal(last().status, 'ready')
  assert.equal(last().retryAfter, null)
})

test('searcher: dispose aborts the request and silences late results', async () => {
  let release
  const { searcher, sched, calls, states } = makeSearcher(() => new Promise((r) => { release = () => r([hit(1)]) }))
  searcher.setQuery('going away'); sched.flush()
  const before = states.length
  searcher.dispose()
  assert.equal(calls[0].signal.aborted, true)
  release(); await tick()
  assert.equal(states.length, before)
})

// ---- excerpt (bold sentence inside the chunk text)
test('excerpt: splits around the highlighted sentence', () => {
  const text = 'Hello everyone. Robots can teach us how to live. Thank you all.'
  assert.deepEqual(excerpt(text, 'Robots can teach us how to live.'),
    { before: 'Hello everyone. ', match: 'Robots can teach us how to live.', after: ' Thank you all.' })
})

test('excerpt: no highlight or one that is not in the text shows the start of the text', () => {
  assert.deepEqual(excerpt('Some text here.', null), { before: '', match: '', after: 'Some text here.' })
  assert.deepEqual(excerpt('Some text here.', 'something else'), { before: '', match: '', after: 'Some text here.' })
  assert.equal(excerpt('x '.repeat(500), null, 50).after.endsWith('…'), true)
})

test('excerpt: tolerates different whitespace and clips long context on word boundaries', () => {
  assert.equal(excerpt('A  b.\n C d.', 'C d.').match, 'C d.')
  const long = `${'before '.repeat(100)}THE MATCH.${' after'.repeat(100)}`
  const e = excerpt(long, 'THE MATCH.', 40)
  assert.ok(e.before.startsWith('…') && e.before.length <= 41)
  assert.ok(e.after.endsWith('…') && e.after.length <= 41)
  assert.equal(e.match, 'THE MATCH.')
  assert.ok(!e.before.slice(1).startsWith('efore')) // cut at a word boundary, not mid-word
})

test('excerpt: a highlight equal to the whole text leaves no context', () => {
  assert.deepEqual(excerpt('Only sentence.', 'Only sentence.'), { before: '', match: 'Only sentence.', after: '' })
})

// ---- keyboard navigation
test('moveActive: arrows clamp at the ends; Up from nothing selected goes to the last result', () => {
  assert.equal(moveActive(-1, 'ArrowDown', 5), 0)
  assert.equal(moveActive(0, 'ArrowDown', 5), 1)
  assert.equal(moveActive(4, 'ArrowDown', 5), 4)
  assert.equal(moveActive(3, 'ArrowUp', 5), 2)
  assert.equal(moveActive(0, 'ArrowUp', 5), 0)
  assert.equal(moveActive(-1, 'ArrowUp', 5), 4)
  assert.equal(moveActive(2, 'ArrowDown', 0), -1)
})

test('scrollTargetFor scrolls only as far as needed', () => {
  assert.equal(scrollTargetFor(100, 300, 150, 40), null) // fully visible
  assert.equal(scrollTargetFor(100, 300, 80, 40), 72) // above: align near the top (pad 8)
  assert.equal(scrollTargetFor(100, 300, 380, 40), 128) // below: align near the bottom
  assert.equal(scrollTargetFor(100, 300, 2, 40), 0) // never negative
})

// ---- presigned URL cache
test('stream URL cache: one fetch per video, shared by concurrent callers', async () => {
  let n = 0
  const cache = createStreamUrlCache(async (id) => `https://s3/${id}?sig=${++n}`)
  const [a, b] = await Promise.all([cache.get('v1'), cache.get('v1')])
  assert.equal(a, b)
  assert.equal(n, 1)
  assert.equal(await cache.get('v1'), a)
  assert.notEqual(await cache.get('v2'), a)
  assert.equal(n, 2)
})

test('stream URL cache: entries expire, can be invalidated, and failures are not cached', async () => {
  let n = 0, clock = 0
  const cache = createStreamUrlCache(async () => `u${++n}`, { ttlMs: 1000, now: () => clock })
  assert.equal(await cache.get('v'), 'u1')
  clock = 999
  assert.equal(await cache.get('v'), 'u1')
  clock = 1000 // exactly at the ttl: expired
  assert.equal(await cache.get('v'), 'u2')
  cache.invalidate('v')
  assert.equal(await cache.get('v'), 'u3')
  cache.clear()
  assert.equal(await cache.get('v'), 'u4')

  let fail = true
  const flaky = createStreamUrlCache(async () => { if (fail) throw new Error('boom'); return 'ok' })
  await assert.rejects(flaky.get('v'), /boom/)
  fail = false
  assert.equal(await flaky.get('v'), 'ok') // the failure was forgotten
})

test('stream URL cache: the default lifetime is shorter than the 1 hour the backend signs for', async () => {
  const { STREAM_URL_TTL_MS } = await import('../.test-build/lib/streamUrls.js')
  assert.ok(STREAM_URL_TTL_MS < 60 * 60 * 1000)
})
