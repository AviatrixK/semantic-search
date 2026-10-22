import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createSearcher } from '../.test-build/lib/searcher.js'
import { KEYWORD_ONLY_NOTE, isKeywordOnly } from '../.test-build/lib/searchFallback.js'
import { createApiClient } from '../.test-build/api/client.js'

const tick = () => new Promise((r) => setTimeout(r, 0))
const hit = { video_id: 'v', title: 'T', start_sec: 1, end_sec: 2, text: 't', score: 0.5, highlight: null }

function searcherFor(run) {
  const states = []
  const searcher = createSearcher({ run, onState: (s) => states.push(s), describe: (e) => ({ message: e.message }), schedule: (fn) => { fn(); return () => {} } })
  return { searcher, last: () => states.at(-1) }
}

test('the note says exactly what the product owner asked for', () => {
  assert.equal(KEYWORD_ONLY_NOTE, 'Showing keyword matches only, as smart search is temporarily unavailable.')
})

test('isKeywordOnly: only the keyword-only value of X-Search-Fallback counts', () => {
  assert.equal(isKeywordOnly(new Headers({ 'X-Search-Fallback': 'keyword-only' })), true)
  assert.equal(isKeywordOnly(new Headers({ 'x-search-fallback': ' Keyword-Only ' })), true) // header names and spacing do not matter
  assert.equal(isKeywordOnly(new Headers({ 'X-Search-Fallback': 'something-else' })), false)
  assert.equal(isKeywordOnly(new Headers()), false)
  assert.equal(isKeywordOnly(null), false)
})

test('the client hands the response headers of a successful request to onResponseHeaders', async () => {
  const fetch = async () => new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json', 'X-Search-Fallback': 'keyword-only' } })
  const client = createApiClient({ fetch })
  let seen = null
  const data = await client.request('/api/search?q=x', { onResponseHeaders: (h) => (seen = isKeywordOnly(h)) })
  assert.deepEqual(data, [])
  assert.equal(seen, true)
})

test('the searcher state carries keywordOnly from the run, and a bare array means normal results', async () => {
  const a = searcherFor(async () => ({ hits: [hit], keywordOnly: true }))
  a.searcher.setQuery('robot')
  await tick()
  assert.equal(a.last().status, 'ready')
  assert.equal(a.last().keywordOnly, true)

  const b = searcherFor(async () => [hit])
  b.searcher.setQuery('robot')
  await tick()
  assert.equal(b.last().keywordOnly, false)
})

test('a later normal answer clears the note', async () => {
  let fallback = true
  const { searcher, last } = searcherFor(async () => ({ hits: [hit], keywordOnly: fallback }))
  searcher.setQuery('robot')
  await tick()
  assert.equal(last().keywordOnly, true)
  fallback = false
  searcher.setQuery('robots')
  await tick()
  assert.equal(last().keywordOnly, false)
})

test('the Search page shows the note above the results list', () => {
  const page = readFileSync(new URL('../src/pages/Search.tsx', import.meta.url), 'utf8')
  const note = page.indexOf('KEYWORD_ONLY_NOTE}')
  assert.ok(note > 0, 'the page renders the note')
  assert.ok(note < page.indexOf('aria-label="Search results"'), 'above the list')
  assert.match(page, /state\.keywordOnly/)
})
