import test from 'node:test'
import assert from 'node:assert/strict'
import {
  DEFAULT_FILTERS, MODE_OPTIONS, activeFilterCount, filtersKey, filtersToParams, foundByBadge, isValidDate, modeOfKey, parseFilters, scoreText, searchUrl,
} from '../.test-build/lib/searchParams.js'
import { createSearcher } from '../.test-build/lib/searcher.js'

const VID = '8f27bfee-4387-448e-bc0f-29cb05d75f80'
const params = (s) => new URLSearchParams(s)
const tick = () => new Promise((r) => setTimeout(r, 0))

test('parseFilters reads mode, video and date from the address, ignoring anything invalid', () => {
  assert.deepEqual(parseFilters(params('')), DEFAULT_FILTERS)
  assert.deepEqual(parseFilters(params(`mode=keyword&video=${VID.toUpperCase()}&after=2024-02-29`)), { mode: 'keyword', videoId: VID, after: '2024-02-29' })
  assert.deepEqual(parseFilters(params('mode=bm25&video=nope&after=2024-02-30')), DEFAULT_FILTERS)
  assert.equal(parseFilters(params('after=2024-2-3')).after, null)
  assert.equal(parseFilters(params('after=2023-02-29')).after, null) // not a leap year
  assert.equal(parseFilters(params(`video=${VID}%27%20OR%201%3D1`)).videoId, null)
})

test('isValidDate accepts real calendar days only', () => {
  for (const ok of ['2024-02-29', '2000-01-01', '2026-12-31']) assert.equal(isValidDate(ok), true, ok)
  for (const bad of ['2024-13-01', '2024-00-10', '2024-04-31', '24-01-01', '2024/01/01', '', 'today']) assert.equal(isValidDate(bad), false, bad)
})

test('filtersToParams leaves out defaults, and parseFilters reads back what it wrote', () => {
  assert.deepEqual(filtersToParams(DEFAULT_FILTERS), {})
  const f = { mode: 'vector', videoId: VID, after: '2025-01-02' }
  assert.deepEqual(filtersToParams(f), { mode: 'vector', video: VID, after: '2025-01-02' })
  assert.deepEqual(parseFilters(new URLSearchParams(filtersToParams(f))), f)
})

test('filtersKey differs for every filter and counts the optional ones', () => {
  const keys = new Set([DEFAULT_FILTERS, { ...DEFAULT_FILTERS, mode: 'vector' }, { ...DEFAULT_FILTERS, videoId: VID }, { ...DEFAULT_FILTERS, after: '2025-01-01' }].map(filtersKey))
  assert.equal(keys.size, 4)
  assert.equal(activeFilterCount(DEFAULT_FILTERS), 0)
  assert.equal(activeFilterCount({ mode: 'keyword', videoId: VID, after: '2025-01-01' }), 2) // the mode is not a filter
})

test('modeOfKey reads the mode back from a filtersKey, defaulting to hybrid', () => {
  for (const mode of ['hybrid', 'vector', 'keyword']) assert.equal(modeOfKey(filtersKey({ ...DEFAULT_FILTERS, mode, videoId: VID })), mode)
  assert.equal(modeOfKey(''), 'hybrid')
  assert.equal(modeOfKey('bm25||'), 'hybrid')
})

test('searchUrl builds the request the backend expects and encodes the query', () => {
  assert.equal(searchUrl('how to pause', 10, DEFAULT_FILTERS), '/api/search?q=how+to+pause&k=10&mode=hybrid')
  const u = new URL(searchUrl('a&b=c #1', 5, { mode: 'keyword', videoId: VID, after: '2025-01-02' }), 'http://x')
  assert.deepEqual([...u.searchParams], [['q', 'a&b=c #1'], ['k', '5'], ['mode', 'keyword'], ['video_id', VID], ['uploaded_after', '2025-01-02']])
})

test('foundByBadge explains which search returned a result', () => {
  assert.equal(foundByBadge(['vector', 'keyword']).label, 'Meaning + words')
  assert.equal(foundByBadge(['vector']).label, 'Meaning')
  assert.equal(foundByBadge(['keyword']).label, 'Exact words')
  for (const none of [undefined, null, [], ['other']]) assert.equal(foundByBadge(none), null)
})

test('scoreText names the bar after what the score means in that mode', () => {
  assert.deepEqual(scoreText('hybrid', 0.5), { label: 'Similarity', title: 'Similarity 0.50' })
  assert.equal(scoreText('keyword', 0.123).title, 'Keyword relevance 0.12')
})

test('every mode option has a label and a hint', () => {
  assert.deepEqual(MODE_OPTIONS.map((m) => m.value), ['hybrid', 'vector', 'keyword'])
  assert.ok(MODE_OPTIONS.every((m) => m.label && m.hint))
})

// ---- the searcher with filters
function setup() {
  const calls = []
  const states = []
  const tasks = []
  const searcher = createSearcher({
    run: async (q, signal, f) => { calls.push({ q, mode: f.mode, video: f.videoId, signal }); return [{ video_id: 'v', title: 'T', start_sec: 1, end_sec: 2, text: 't', score: 0.5, highlight: null }] },
    onState: (s) => states.push(s),
    describe: (e) => ({ message: e.message }),
    schedule: (fn) => { const t = { fn, off: false }; tasks.push(t); return () => { t.off = true } },
  })
  return { searcher, calls, states, last: () => states.at(-1), flush: () => tasks.splice(0).filter((t) => !t.off).forEach((t) => t.fn()) }
}

test('searcher: searches with the starting filters and records which filters the results belong to', async () => {
  const s = setup()
  s.searcher.setQuery('pauses')
  s.flush()
  await tick()
  assert.deepEqual(s.calls.map((c) => c.mode), ['hybrid'])
  assert.equal(s.last().filtersKey, filtersKey(DEFAULT_FILTERS))
})

test('searcher: changing a filter re-runs the current query at once, without waiting for the debounce', async () => {
  const s = setup()
  s.searcher.setQuery('pauses')
  s.flush()
  await tick()
  s.searcher.setFilters({ ...DEFAULT_FILTERS, mode: 'keyword' })
  assert.equal(s.last().status, 'loading') // immediately, no timer involved
  await tick()
  assert.deepEqual(s.calls.map((c) => c.mode), ['hybrid', 'keyword'])
  assert.equal(s.last().filtersKey, filtersKey({ ...DEFAULT_FILTERS, mode: 'keyword' }))
})

test('searcher: the same filters again do nothing; the same query + same filters is not searched twice', async () => {
  const s = setup()
  s.searcher.setQuery('pauses')
  s.flush()
  await tick()
  s.searcher.setFilters({ ...DEFAULT_FILTERS })
  s.searcher.submit()
  await tick()
  assert.equal(s.calls.length, 1)
})

test('searcher: filters changed before any query wait for the first query and are used by it', async () => {
  const s = setup()
  s.searcher.setFilters({ ...DEFAULT_FILTERS, videoId: VID })
  assert.equal(s.calls.length, 0)
  s.searcher.setQuery('pauses')
  s.flush()
  await tick()
  assert.deepEqual(s.calls.map((c) => c.video), [VID])
})

test('searcher: a slow response for the old filters never overwrites the new ones', async () => {
  const resolvers = []
  const states = []
  const searcher = createSearcher({
    run: (q, signal, f) => new Promise((resolve) => resolvers.push(() => resolve([{ video_id: f.mode, title: f.mode, start_sec: 0, end_sec: 1, text: '', score: 1, highlight: null }]))),
    onState: (s) => states.push(s),
    describe: (e) => ({ message: String(e) }),
    schedule: (fn) => { fn(); return () => {} },
  })
  searcher.setQuery('pauses') // hybrid request #1 starts
  searcher.setFilters({ ...DEFAULT_FILTERS, mode: 'vector' }) // vector request #2 supersedes it
  resolvers[1]()
  await tick()
  resolvers[0]() // the old one finishes last
  await tick()
  assert.equal(states.at(-1).hits[0].title, 'vector')
  assert.equal(states.at(-1).filtersKey, filtersKey({ ...DEFAULT_FILTERS, mode: 'vector' }))
})
