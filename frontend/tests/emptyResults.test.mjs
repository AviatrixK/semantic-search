import test from 'node:test'
import assert from 'node:assert/strict'
import { emptyResults } from '../.test-build/lib/emptyResults.js'

test('names the query in straight quotes', () => {
  assert.equal(emptyResults('public speaking', 0).title, "No results for 'public speaking'")
})

test('no processing note when nothing is processing', () => {
  assert.equal(emptyResults('x y', 0).processingNote, null)
})

test('plural and singular processing notes', () => {
  assert.equal(emptyResults('q', 3).processingNote, '3 videos are still processing and are not searchable yet.')
  assert.equal(emptyResults('q', 1).processingNote, '1 video is still processing and is not searchable yet.')
})

test('bad counts never produce a note', () => {
  assert.equal(emptyResults('q', -2).processingNote, null)
  assert.equal(emptyResults('q', Number.NaN).processingNote, null)
})
