import test from 'node:test'
import assert from 'node:assert/strict'
import { chooseSnippet, languageLabel, markSegments, queryTerms, scriptOf } from '../.test-build/lib/snippet.js'

const TAMIL = 'ஒரு நல்ல ஆசிரியர் மாணவர்களுக்கு கற்றுக்கொடுக்க வேண்டும்'
const EN = 'A good teacher must teach the students.'
const tamilHit = { text: TAMIL, text_en: EN, language: 'ta-IN', highlight: null }

test('script detection', () => {
  assert.equal(scriptOf('public speaking'), 'latin')
  assert.equal(scriptOf(TAMIL), 'tamil')
  assert.equal(scriptOf('नमस्ते दोस्तों'), 'devanagari')
  assert.equal(scriptOf('123 !!'), '')
})

test('English query on a Tamil chunk: English first, original below', () => {
  const c = chooseSnippet(tamilHit, 'good teacher')
  assert.deepEqual(c.primary, { kind: 'english', text: EN })
  assert.deepEqual(c.secondary, { kind: 'original', text: TAMIL })
})

test('Hinglish (Latin) query on a Hindi chunk: English first', () => {
  const hit = { text: 'शिक्षक को धैर्य रखना चाहिए', text_en: 'A teacher should be patient.', language: 'hi-IN' }
  assert.equal(chooseSnippet(hit, 'shikshak ko dhairya').primary.kind, 'english')
})

test('query in the chunk\'s own script: original first, English below', () => {
  const c = chooseSnippet(tamilHit, 'நல்ல ஆசிரியர்')
  assert.equal(c.primary.kind, 'original')
  assert.deepEqual(c.secondary, { kind: 'english', text: EN })
})

test('Hinglish query on a romanised (Latin-script) non-English chunk: original first', () => {
  const hit = { text: 'shikshak ko dhairya rakhna chahiye', text_en: 'A teacher should be patient.', language: 'hi-IN' }
  const c = chooseSnippet(hit, 'dhairya')
  assert.equal(c.primary.kind, 'original')
  assert.equal(c.secondary.kind, 'english')
})

test('the toggle swaps the two versions', () => {
  const c = chooseSnippet(tamilHit, 'good teacher', true)
  assert.equal(c.primary.kind, 'original')
  assert.equal(c.secondary.kind, 'english')
})

test('missing or blank text_en: original only, no error', () => {
  for (const text_en of [undefined, null, '', '   ']) {
    const c = chooseSnippet({ ...tamilHit, text_en }, 'good teacher')
    assert.deepEqual(c, { primary: { kind: 'original', text: TAMIL }, secondary: null })
    assert.equal(chooseSnippet({ ...tamilHit, text_en }, 'x', true).secondary, null)
  }
})

test('English video: original only, even if a copy exists', () => {
  const hit = { text: EN, text_en: EN, language: 'en' }
  assert.equal(chooseSnippet(hit, 'teacher').secondary, null)
  assert.equal(chooseSnippet({ ...hit, language: 'en-IN', text_en: 'Another copy.' }, 'teacher').secondary, null)
})

test('unknown language: decided by the chunk\'s script', () => {
  assert.equal(chooseSnippet({ text: TAMIL, text_en: EN }, 'teacher').primary.kind, 'english')
  assert.equal(chooseSnippet({ text: EN, text_en: 'Same, reworded.' }, 'teacher').secondary, null)
})

test('language badge: Sarvam and Whisper codes', () => {
  assert.equal(languageLabel('ta-IN'), 'Tamil')
  assert.equal(languageLabel('ta'), 'Tamil')
  assert.equal(languageLabel('en'), 'English')
  assert.equal(languageLabel(null), null)
  assert.equal(languageLabel(''), null)
})

test('query terms drop stop words and one-letter words', () => {
  assert.deepEqual(queryTerms('What is a Good teacher?'), ['good', 'teacher'])
})

const joined = (segs) => segs.map((s) => (s.hit ? `[${s.text}]` : s.text)).join('')

test('English text is highlighted by stem', () => {
  const segs = markSegments('Teachers love speaking about the Teacher role.', queryTerms('teacher speak'), { kind: 'english' })
  assert.equal(joined(segs), '[Teachers] love [speaking] about the [Teacher] role.')
})

test('original text: exact words only; a Latin query leaves Tamil unmarked unless the best sentence is known', () => {
  assert.equal(joined(markSegments(TAMIL, queryTerms('good teacher'), { kind: 'original' })), TAMIL)
  const first = 'ஒரு நல்ல ஆசிரியர்'
  assert.equal(joined(markSegments(TAMIL, ['x'], { kind: 'original', highlight: first })), `[${first}] மாணவர்களுக்கு கற்றுக்கொடுக்க வேண்டும்`)
  assert.equal(joined(markSegments(TAMIL, queryTerms('ஆசிரியர்'), { kind: 'original' })), 'ஒரு நல்ல [ஆசிரியர்] மாணவர்களுக்கு கற்றுக்கொடுக்க வேண்டும்')
})

test('long text is clipped around the first match', () => {
  const text = `${'filler words '.repeat(60)}the teacher arrives ${'more filler '.repeat(60)}`
  const out = joined(markSegments(text, ['teacher'], { kind: 'english', max: 120 }))
  assert.ok(out.includes('[teacher]') && out.startsWith('…') && out.endsWith('…') && out.length < 160)
})
