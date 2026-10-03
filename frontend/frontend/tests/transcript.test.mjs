import test from 'node:test'
import assert from 'node:assert/strict'
import { activeLineIndex, buildTranscript, splitSentences, trimOverlap } from '../.test-build/lib/transcript.js'
import { parseTimeParam, watchPath } from '../.test-build/lib/timeParam.js'

// Real chunks 0-2 of video 8f27bfee-4387-448e-bc0f-29cb05d75f80: ~30 s windows that overlap by 9-11 s.
const REAL = [
  { idx: 0, start_sec: 0, end_sec: 28, text: "Every time you open your mouth and you speak, people unconsciously categorize you into one of these three categories. Are you an awkward communicator? Are you a pretty good communicator? Or are you a natural communicator? And after 15 years as a communication coach, training Fortune 500 companies and speaking to millions of people around the world, I've discovered that you can tell which level someone sits at based on how they use their voice, how they use their body language, and how they use their words." },
  { idx: 1, start_sec: 18.6, end_sec: 48, text: "I've discovered that you can tell which level someone sits at based on how they use their voice, how they use their body language, and how they use their words. I'm going to explain the exact behaviors at each level, so by the end you know which category you're in, and you know how to progress from one category to the next. To make this even more fun, we're going to put these people you see here on the table into one of these categories, and see if you can guess where they sit on the board." },
  { idx: 2, start_sec: 37.8, end_sec: 67, text: "To make this even more fun, we're going to put these people you see here on the table into one of these categories, and see if you can guess where they sit on the board. Well, why is my face there then? Well Craig, it's so we can demonstrate to the audience there's a level lower than rookie. Let's start with the rookie communicator, and chances are most people will belong in this category without even realizing it, because they don't know what they're doing wrong." },
]
const count = (haystack, needle) => haystack.split(needle).length - 1

test('trimOverlap removes the start of `next` that repeats the end of `prev`', () => {
  const r = trimOverlap(REAL[0].text, REAL[1].text)
  assert.ok(r.removed > 100)
  assert.ok(r.text.startsWith("I'm going to explain the exact behaviors"))
})

test('trimOverlap only matches on word boundaries and returns the text untouched without a match', () => {
  assert.deepEqual(trimOverlap('say hello world', 'world is big'), { text: 'is big', removed: 5 })
  assert.deepEqual(trimOverlap('say hello', 'lo there'), { text: 'lo there', removed: 0 }) // "lo" is the tail of "hello"
  assert.deepEqual(trimOverlap('abc', 'xyz'), { text: 'xyz', removed: 0 })
  assert.deepEqual(trimOverlap('same text', 'same text'), { text: '', removed: 9 })
})

test('buildTranscript: no sentence appears twice although the chunks overlap', () => {
  const lines = buildTranscript(REAL)
  const all = lines.map((l) => l.text).join(' ')
  assert.equal(count(all, "I've discovered that you can tell"), 1)
  assert.equal(count(all, 'To make this even more fun'), 1)
  assert.equal(count(all, 'Well, why is my face there then?'), 1)
  assert.equal(lines.length, 5 + 2 + 3) // chunk 0: 5 sentences; chunk 1 adds 2 new ones; chunk 2 adds 3 new ones
})

test('buildTranscript: times are ordered, start at 0, and new material begins where the previous chunk ended', () => {
  const lines = buildTranscript(REAL)
  assert.equal(lines[0].start, 0)
  for (let i = 1; i < lines.length; i++) assert.ok(lines[i].start >= lines[i - 1].start, `line ${i} goes back in time`)
  const firstOfChunk1 = lines.find((l) => l.text.startsWith("I'm going to explain"))
  assert.equal(firstOfChunk1.start, 28) // chunk 0 ended at 28
  const firstOfChunk2 = lines.find((l) => l.text.startsWith('Well, why is my face'))
  assert.equal(firstOfChunk2.start, 48) // chunk 1 ended at 48
  for (let i = 0; i < lines.length - 1; i++) assert.equal(lines[i].end, lines[i + 1].start)
  assert.equal(lines.at(-1).end, 67)
  assert.ok(lines.every((l) => l.start >= 0 && l.start <= 67))
})

test('buildTranscript: sentence starts inside a chunk are interpolated by text position', () => {
  const lines = buildTranscript([{ idx: 0, start_sec: 10, end_sec: 30, text: 'Aaaa aaaa. Bbbb bbbb. Cccc cccc.' }])
  assert.deepEqual(lines.map((l) => l.text), ['Aaaa aaaa.', 'Bbbb bbbb.', 'Cccc cccc.'])
  assert.equal(lines[0].start, 10)
  assert.ok(lines[1].start > 10 && lines[1].start < lines[2].start && lines[2].start < 30)
})

test('buildTranscript: chunks that do not overlap in time are left alone, even with repeated words', () => {
  const lines = buildTranscript([
    { idx: 0, start_sec: 0, end_sec: 30, text: 'We begin here. Thank you.' },
    { idx: 1, start_sec: 30, end_sec: 60, text: 'Thank you. And we continue.' },
  ])
  assert.deepEqual(lines.map((l) => l.text), ['We begin here.', 'Thank you.', 'Thank you.', 'And we continue.'])
})

test('buildTranscript: overlap in time but no matching text keeps the whole chunk; unsorted input is sorted', () => {
  const lines = buildTranscript([
    { idx: 1, start_sec: 20, end_sec: 50, text: 'Different words entirely.' },
    { idx: 0, start_sec: 0, end_sec: 30, text: 'Original words.' },
  ])
  assert.deepEqual(lines.map((l) => [l.text, l.start]), [['Original words.', 0], ['Different words entirely.', 20]])
})

test('buildTranscript: empty input and fully repeated chunks', () => {
  assert.deepEqual(buildTranscript([]), [])
  const lines = buildTranscript([
    { idx: 0, start_sec: 0, end_sec: 10, text: 'Only line.' },
    { idx: 1, start_sec: 5, end_sec: 10, text: 'Only line.' },
  ])
  assert.deepEqual(lines.map((l) => l.text), ['Only line.'])
})

test('activeLineIndex: the last line that has started', () => {
  const lines = [{ start: 0 }, { start: 5 }, { start: 9.5 }, { start: 20 }]
  assert.equal(activeLineIndex([], 3), -1)
  assert.equal(activeLineIndex([{ start: 4 }], 3.9), -1) // before the first line
  assert.equal(activeLineIndex(lines, 0), 0)
  assert.equal(activeLineIndex(lines, 4.99), 0)
  assert.equal(activeLineIndex(lines, 5), 1)
  assert.equal(activeLineIndex(lines, 19.99), 2)
  assert.equal(activeLineIndex(lines, 9999), 3)
})

test('splitSentences: abbreviations, initials, offsets', () => {
  assert.deepEqual(splitSentences('Dr. Smith spoke. Then J. K. Rowling did. Why? Because.').map((s) => s.text),
    ['Dr. Smith spoke.', 'Then J. K. Rowling did.', 'Why?', 'Because.'])
  const s = splitSentences('One. Two.')
  assert.deepEqual(s.map((x) => x.offset), [0, 5])
  assert.deepEqual(splitSentences('no end punctuation here').map((x) => x.text), ['no end punctuation here'])
  assert.deepEqual(splitSentences('Hmm... okay then. Fine.').map((x) => x.text), ['Hmm... okay then.', 'Fine.'])
})

// ---- deep links
test('parseTimeParam: seconds, clock times and youtube-style times', () => {
  for (const [input, expected] of [
    ['123', 123], ['12.5', 12.5], ['0', 0], ['2:03', 123], ['02:03', 123], ['1:02:03', 3723], ['90s', 90],
    ['1m30s', 90], ['1h2m3s', 3723], ['2m', 120], ['1H', 3600], [' 45 ', 45],
  ]) assert.equal(parseTimeParam(input), expected, input)
})

test('parseTimeParam rejects junk, negatives and absurd values', () => {
  for (const input of [null, undefined, '', ' ', 'abc', '-5', '1:75', '1:2:99', '12s34', '1e3', 'NaN', '99999999', '1.5.2', 'm', 'h1']) {
    assert.equal(parseTimeParam(input), null, String(input))
  }
})

test('watchPath builds deep links with whole seconds', () => {
  assert.equal(watchPath('abc'), '/watch/abc')
  assert.equal(watchPath('abc', 123.9), '/watch/abc?t=123')
  assert.equal(watchPath('abc', -4), '/watch/abc?t=0')
  assert.equal(watchPath('a b/c', 5), '/watch/a%20b%2Fc?t=5')
})
