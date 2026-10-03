import test from 'node:test'
import assert from 'node:assert/strict'
import { hasPendingUploads, initialQueue, nextToStart, queueReducer } from '../.test-build/lib/uploadQueue.js'

const file = (name, size = 1000) => ({ name, size })
const run = (...actions) => actions.reduce((s, a) => queueReducer(s, a), initialQueue())
const enqueue = (...names) => ({ type: 'enqueue', entries: names.map((n) => ({ id: n, file: file(n), title: n.toUpperCase() })) })
const phases = (s) => s.items.map((i) => `${i.id}:${i.phase}`)

test('enqueue adds queued items in order', () => {
  const s = run(enqueue('a', 'b'))
  assert.deepEqual(phases(s), ['a:queued', 'b:queued'])
  assert.equal(s.items[0].title, 'A')
  assert.equal(s.items[0].total, 1000)
})

test('files upload one at a time, in order', () => {
  let s = run(enqueue('a', 'b', 'c'))
  assert.equal(nextToStart(s), 'a')
  s = run(...[], { type: 'start', id: 'a' }) // fresh state for clarity below
  s = queueReducer(run(enqueue('a', 'b', 'c')), { type: 'start', id: 'a' })
  assert.equal(nextToStart(s), undefined) // a is uploading: b must wait
  s = queueReducer(s, { type: 'uploaded', id: 'a', videoId: 'v1', jobId: 'j1' })
  assert.equal(nextToStart(s), 'b') // a is only processing now, so b may start
  s = queueReducer(s, { type: 'start', id: 'b' })
  s = queueReducer(s, { type: 'failed', id: 'b', message: 'nope' })
  assert.equal(nextToStart(s), 'c') // a failure also frees the slot
})

test('start is idempotent and only applies to queued items', () => {
  const s1 = run(enqueue('a'), { type: 'start', id: 'a' })
  const s2 = queueReducer(s1, { type: 'start', id: 'a' })
  assert.equal(s2, s1) // same object: nothing re-renders
  assert.equal(queueReducer(s1, { type: 'start', id: 'missing' }), s1)
})

test('progress only updates an uploading item', () => {
  const queued = run(enqueue('a'))
  assert.equal(queueReducer(queued, { type: 'progress', id: 'a', loaded: 5, total: 10 }), queued)
  const s = run(enqueue('a'), { type: 'start', id: 'a' }, { type: 'progress', id: 'a', loaded: 400, total: 1000 })
  assert.equal(s.items[0].loaded, 400)
})

test('uploaded -> processing: keeps ids, drops the File, shows 100%', () => {
  const s = run(enqueue('a'), { type: 'start', id: 'a' }, { type: 'progress', id: 'a', loaded: 400, total: 1000 },
    { type: 'uploaded', id: 'a', videoId: 'v1', jobId: 'j1' })
  const item = s.items[0]
  assert.equal(item.phase, 'processing')
  assert.equal(item.file, null)
  assert.equal(item.loaded, item.total)
  assert.deepEqual([item.videoId, item.jobId], ['v1', 'j1'])
})

test('failed: from queued or uploading, records the message and drops the File', () => {
  const s = run(enqueue('a', 'b'), { type: 'start', id: 'a' }, { type: 'failed', id: 'a', message: 'too big' },
    { type: 'failed', id: 'b', message: 'unsupported' })
  assert.deepEqual(s.items.map((i) => [i.phase, i.error, i.file]), [['error', 'too big', null], ['error', 'unsupported', null]])
})

test('a late "uploaded" or "failed" for a removed (cancelled) item changes nothing', () => {
  const s = run(enqueue('a'), { type: 'start', id: 'a' }, { type: 'remove', id: 'a' })
  assert.deepEqual(s.items, [])
  assert.equal(queueReducer(s, { type: 'uploaded', id: 'a', videoId: 'v', jobId: 'j' }), s)
  assert.equal(queueReducer(s, { type: 'failed', id: 'a', message: 'x' }), s)
})

test('remove and removeByVideo', () => {
  let s = run(enqueue('a', 'b'), { type: 'start', id: 'a' }, { type: 'uploaded', id: 'a', videoId: 'v1', jobId: 'j1' })
  s = queueReducer(s, { type: 'removeByVideo', videoId: 'v1' })
  assert.deepEqual(phases(s), ['b:queued'])
  s = queueReducer(s, { type: 'remove', id: 'b' })
  assert.deepEqual(s.items, [])
})

test('track adds a processing card for a job started elsewhere (reprocess)', () => {
  const s = run({ type: 'track', id: 't1', videoId: 'v9', jobId: 'j9', title: 'Old talk' })
  assert.deepEqual(phases(s), ['t1:processing'])
  assert.equal(s.items[0].jobId, 'j9')
  assert.equal(nextToStart(s), undefined)
})

test('hasPendingUploads is true only while files wait or upload', () => {
  assert.equal(hasPendingUploads(initialQueue()), false)
  let s = run(enqueue('a'))
  assert.equal(hasPendingUploads(s), true)
  s = run(enqueue('a'), { type: 'start', id: 'a' })
  assert.equal(hasPendingUploads(s), true)
  s = queueReducer(s, { type: 'uploaded', id: 'a', videoId: 'v', jobId: 'j' })
  assert.equal(hasPendingUploads(s), false) // processing continues server-side; leaving the page is safe
})
