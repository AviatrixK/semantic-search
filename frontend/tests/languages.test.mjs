import test from 'node:test'
import assert from 'node:assert/strict'
import { AUTO_LANGUAGE, LANGUAGES, isLanguageChoice } from '../.test-build/lib/languages.js'
import { buildUploadForm } from '../.test-build/lib/uploadValidation.js'
import { initialQueue, queueReducer } from '../.test-build/lib/uploadQueue.js'

const file = (name = 'talk.mp4') => new File([new Uint8Array(4)], name, { type: 'video/mp4' })

test('the dropdown has the 23 supported languages, Hindi and English first, no duplicates', () => {
  assert.equal(LANGUAGES.length, 23)
  assert.deepEqual(LANGUAGES.slice(0, 2).map((l) => l.code), ['hi-IN', 'en-IN'])
  assert.equal(new Set(LANGUAGES.map((l) => l.code)).size, 23)
  assert.ok(LANGUAGES.every((l) => /^[a-z]{2,3}-IN$/.test(l.code) && l.label.length > 2))
})

test('only "" (automatic) or a listed code is a valid choice', () => {
  assert.equal(AUTO_LANGUAGE, '')
  for (const ok of ['', 'hi-IN', 'doi-IN']) assert.equal(isLanguageChoice(ok), true, ok)
  for (const bad of ['hi', 'klingon', 'HI-IN', ' hi-IN', null, undefined, 5, '<script>']) assert.equal(isLanguageChoice(bad), false, String(bad))
})

test('buildUploadForm sends the language only when one was chosen', () => {
  assert.equal(buildUploadForm(file(), 'Talk').has('language'), false) // automatic: the server detects it
  assert.equal(buildUploadForm(file(), 'Talk', '').has('language'), false)
  assert.equal(buildUploadForm(file(), 'Talk', 'ta-IN').get('language'), 'ta-IN')
  assert.equal(buildUploadForm(file(), 'Talk', 'klingon').has('language'), false) // never send something the server would reject
})

test('the chosen language travels with the queued upload, and defaults to automatic', () => {
  const entries = [{ id: 'a', file: file('a.mp4'), title: 'A', language: 'hi-IN' }, { id: 'b', file: file('b.mp4'), title: 'B' }]
  const s = queueReducer(initialQueue(), { type: 'enqueue', entries })
  assert.deepEqual(s.items.map((i) => i.language), ['hi-IN', ''])
  const t = queueReducer(s, { type: 'track', id: 'c', videoId: 'v', jobId: 'j', title: 'C' })
  assert.equal(t.items[2].language, '') // a reprocess card has no upload language of its own
})
