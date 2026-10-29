import test from 'node:test'
import assert from 'node:assert/strict'
import { activeLineIndex, buildTranscript, dropRepeatedStart, seekRequest, splitSentences } from '../.test-build/lib/transcript.js'
import {
  TRANSCRIPT_LANG_KEY,
  initialTranscriptLang,
  linkLangForQuery,
  parseTranscriptLang,
  readSavedTranscriptLang,
  saveTranscriptLang,
} from '../.test-build/lib/transcriptLang.js'
import { isNonEnglishVideo } from '../.test-build/lib/snippet.js'
import { watchPath } from '../.test-build/lib/timeParam.js'

// Three Tamil sentences per chunk; chunk 1 starts with exactly the text chunk 0 ends with (the real overlap), but each chunk was
// translated on its own, so the repeated sentence is worded differently in English.
const T = [
  'வணக்கம் நண்பர்களே.',
  'இன்று நாம் ஆசிரியர் பற்றி பேசுவோம்.',
  'ஒரு நல்ல ஆசிரியர் பொறுமையாக இருக்க வேண்டும்.',
  'மாணவர்களை ஊக்குவிக்க வேண்டும்.',
  'நன்றி நண்பர்களே.',
]
const CHUNKS = [
  {
    idx: 0, start_sec: 0, end_sec: 30,
    text: `${T[0]} ${T[1]} ${T[2]}`,
    text_en: 'Hello friends. Today we will talk about teachers. A good teacher must be patient.',
    text_roman: 'Vanakkam nanbargale. Indru naam aasiriyar patri pesuvom. Oru nalla aasiriyar porumaiyaaga irukka vendum.',
  },
  {
    idx: 1, start_sec: 25, end_sec: 55,
    text: `${T[2]} ${T[3]} ${T[4]}`,
    text_en: 'A good teacher needs patience. Students must be encouraged. Thank you friends.',
    text_roman: 'Oru nalla aasiriyar porumai vendum. Maanavarkalai ookkuvikka vendum. Nandri nanbargale.',
  },
]
const texts = (lines) => lines.map((l) => l.text)

test('original text: Tamil sentences are split and the overlap is not repeated', () => {
  const lines = buildTranscript(CHUNKS)
  assert.deepEqual(texts(lines), T)
  assert.ok(lines.every((l) => l.pending === false))
})

test('English: no sentence appears twice although the repeated part was translated differently', () => {
  const lines = buildTranscript(CHUNKS, 'en')
  assert.deepEqual(texts(lines), [
    'Hello friends.',
    'Today we will talk about teachers.',
    'A good teacher must be patient.',
    'Students must be encouraged.',
    'Thank you friends.',
  ])
  assert.ok(lines.every((l) => !l.pending))
})

test('English: a repeat translated identically is found exactly', () => {
  const same = [CHUNKS[0], { ...CHUNKS[1], text_en: 'A good teacher must be patient. Students must be encouraged. Thank you friends.' }]
  assert.deepEqual(texts(buildTranscript(same, 'en')).filter((t) => t.startsWith('A good teacher')), ['A good teacher must be patient.'])
})

test('romanised text gets the same treatment', () => {
  const lines = texts(buildTranscript(CHUNKS, 'roman'))
  assert.equal(lines.filter((t) => /nalla aasiriyar/i.test(t)).length, 1)
  assert.equal(lines.length, 5)
})

test('a single-sentence translation (no capitals or full stops) is cut by position, never repeated whole', () => {
  const words = (n, p) => Array.from({ length: n }, (_, i) => `${p}${i}`).join(' ')
  const a = { idx: 0, start_sec: 0, end_sec: 30, text: 'aa bb cc dd.', text_roman: words(8, 'x') }
  const b = { idx: 1, start_sec: 25, end_sec: 55, text: 'cc dd. ee ff.', text_roman: words(8, 'y') }
  const roman = texts(buildTranscript([a, b], 'roman')).join(' ')
  // half of chunk 1 was repeat: its first words are dropped
  assert.ok(roman.includes('y4') && !roman.includes('y0') && roman.endsWith('y7'))
})

test('pending: a chunk without a translation shows its original lines, flagged; others are not flagged', () => {
  const lines = buildTranscript([CHUNKS[0], { ...CHUNKS[1], text_en: null }], 'en')
  assert.deepEqual(lines.filter((l) => !l.pending).map((l) => l.text).length, 3)
  const pending = lines.filter((l) => l.pending)
  assert.deepEqual(texts(pending), [T[3], T[4]]) // the original, with the overlap already removed
  assert.deepEqual(texts(buildTranscript([{ ...CHUNKS[0], text_en: '   ' }], 'en')), T.slice(0, 3))
  assert.ok(buildTranscript([{ ...CHUNKS[0], text_roman: undefined }], 'roman').every((l) => l.pending))
})

test('lines of every language are ordered, start at the chunk start, and new material starts after the previous chunk ended', () => {
  for (const lang of ['original', 'en', 'roman']) {
    const lines = buildTranscript(CHUNKS, lang)
    assert.equal(lines[0].start, 0, lang)
    for (let i = 1; i < lines.length; i++) assert.ok(lines[i].start >= lines[i - 1].start, lang)
    assert.ok(lines.filter((l) => l.id.startsWith('1.')).every((l) => l.start >= 30), lang)
  }
})

test('the current line is found in every language', () => {
  for (const lang of ['original', 'en', 'roman']) {
    const lines = buildTranscript(CHUNKS, lang)
    for (const t of [0, 12, 29, 31, 40, 54]) {
      const i = activeLineIndex(lines, t)
      assert.ok(i >= 0 && lines[i].start <= t && (i === lines.length - 1 || lines[i + 1].start > t), `${lang} @${t}`)
    }
    assert.ok(lines[activeLineIndex(lines, 40)].id.startsWith('1.'), lang) // second 40 is spoken in chunk 1 whatever the language
  }
})

test('clicking a translated line seeks to that line', () => {
  const lines = buildTranscript(CHUNKS, 'en')
  const line = lines[3]
  assert.deepEqual(seekRequest('vid-1', line, 7), { videoId: 'vid-1', t: line.start, play: true, nonce: 7 })
  assert.ok(line.start >= 30 && line.start < 55)
  assert.equal(seekRequest('v', buildTranscript(CHUNKS, 'roman')[0], 1).t, 0)
})

test('dropRepeatedStart: nothing repeated keeps the text; everything repeated drops it', () => {
  assert.equal(dropRepeatedStart('One. Two.', null, 0), 'One. Two.')
  assert.equal(dropRepeatedStart('One. Two.', 'Zero.', 1), '')
})

test('Hindi sentences end at the danda, even without capitals', () => {
  const s = splitSentences('नमस्ते दोस्तों। आज हम बात करेंगे। धन्यवाद')
  assert.deepEqual(s.map((x) => x.text), ['नमस्ते दोस्तों।', 'आज हम बात करेंगे।', 'धन्यवाद'])
  assert.deepEqual(splitSentences('Hmm... okay then. Fine.').map((x) => x.text), ['Hmm... okay then.', 'Fine.'])
})

// ---------------------------------------------------------------- choosing the language
test('parseTranscriptLang accepts only the three languages', () => {
  assert.equal(parseTranscriptLang('en'), 'en')
  assert.equal(parseTranscriptLang('roman'), 'roman')
  assert.equal(parseTranscriptLang('original'), 'original')
  for (const bad of ['EN', 'hindi', '', null, undefined, 5]) assert.equal(parseTranscriptLang(bad), null)
})

test('default rule: ?lang=en opens English, otherwise Original, and a saved choice wins', () => {
  const base = { toggleShown: true, saved: null, urlLang: null }
  assert.equal(initialTranscriptLang(base), 'original')
  assert.equal(initialTranscriptLang({ ...base, urlLang: 'en' }), 'en')
  assert.equal(initialTranscriptLang({ ...base, urlLang: 'klingon' }), 'original')
  assert.equal(initialTranscriptLang({ ...base, saved: 'roman', urlLang: 'en' }), 'roman')
  assert.equal(initialTranscriptLang({ ...base, saved: 'original', urlLang: 'en' }), 'original')
})

test('an English video never switches language, whatever was saved or asked for', () => {
  assert.equal(initialTranscriptLang({ toggleShown: false, saved: 'roman', urlLang: 'en' }), 'original')
})

test('search links add ?lang=en only for a Latin-script query on a non-English video', () => {
  assert.equal(linkLangForQuery('good teacher', true), 'en')
  assert.equal(linkLangForQuery('shikshak ko dhairya', true), 'en') // Hinglish
  assert.equal(linkLangForQuery('நல்ல ஆசிரியர்', true), undefined)
  assert.equal(linkLangForQuery('शिक्षक', true), undefined)
  assert.equal(linkLangForQuery('good teacher', false), undefined)
  assert.equal(watchPath('abc', 65, 'en'), '/watch/abc?t=65&lang=en')
  assert.equal(watchPath('abc', undefined, 'en'), '/watch/abc?lang=en')
  assert.equal(watchPath('abc', 65, undefined), '/watch/abc?t=65')
})

test('which videos are non-English: by language code, else by the script of the text', () => {
  assert.equal(isNonEnglishVideo('ta-IN', 'anything'), true)
  assert.equal(isNonEnglishVideo('en', 'anything'), false)
  assert.equal(isNonEnglishVideo('en-IN', 'anything'), false)
  assert.equal(isNonEnglishVideo(null, T[0]), true)
  assert.equal(isNonEnglishVideo(null, 'Hello there'), false)
})

// ---------------------------------------------------------------- localStorage
const memory = () => {
  const m = new Map()
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => void m.set(k, String(v)), m }
}
const blocked = {
  getItem() { throw new DOMException('denied', 'SecurityError') },
  setItem() { throw new DOMException('quota', 'QuotaExceededError') },
}

test('the choice is remembered and read back', () => {
  const s = memory()
  assert.equal(readSavedTranscriptLang(s), null)
  assert.equal(saveTranscriptLang('roman', s), true)
  assert.equal(s.m.get(TRANSCRIPT_LANG_KEY), 'roman')
  assert.equal(readSavedTranscriptLang(s), 'roman')
})

test('a stored value that is not a language is ignored', () => {
  const s = memory()
  s.setItem(TRANSCRIPT_LANG_KEY, '<script>')
  assert.equal(readSavedTranscriptLang(s), null)
})

test('localStorage that throws, or is missing, never breaks the page', () => {
  assert.equal(readSavedTranscriptLang(blocked), null)
  assert.equal(saveTranscriptLang('en', blocked), false)
  assert.doesNotThrow(() => readSavedTranscriptLang(undefined))
  assert.equal(saveTranscriptLang('en', undefined), false) // Node has no localStorage: the default accessor must not throw either
})
