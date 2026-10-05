import test from 'node:test'
import assert from 'node:assert/strict'
import { createSseParser } from '../.test-build/lib/sse.js'

function parse(chunks, { end = true } = {}) {
  const events = []
  const p = createSseParser((e) => events.push(e))
  for (const c of chunks) p.push(c)
  if (end) p.end()
  return events
}

const STREAM =
  'event: route\ndata: {"route":"agent"}\n\n' +
  ': keep-alive\n\n' +
  'event: step_start\ndata: {"step":1}\n\n' +
  'event: answer\ndata: {"answer":"Hi [1]"}\n\n' +
  'event: done\ndata: {}\n\n'
const EXPECTED = [
  { event: 'route', data: '{"route":"agent"}' }, { event: 'step_start', data: '{"step":1}' },
  { event: 'answer', data: '{"answer":"Hi [1]"}' }, { event: 'done', data: '{}' },
]

test('a plain stream gives its events in order and ignores comment keep-alives', () => {
  assert.deepEqual(parse([STREAM]), EXPECTED)
})

test('the same stream cut at EVERY possible position gives identical events (chunks can split anywhere)', () => {
  for (let i = 0; i <= STREAM.length; i++) {
    assert.deepEqual(parse([STREAM.slice(0, i), STREAM.slice(i)]), EXPECTED, `cut at ${i}`)
  }
  // ... and one character at a time
  assert.deepEqual(parse([...STREAM]), EXPECTED)
})

test('CRLF and lone CR line endings work, even when "\\r" and "\\n" arrive in different chunks', () => {
  const crlf = STREAM.replaceAll('\n', '\r\n')
  assert.deepEqual(parse([crlf]), EXPECTED)
  assert.deepEqual(parse([STREAM.replaceAll('\n', '\r')]), EXPECTED)
  for (let i = 0; i <= crlf.length; i++) assert.deepEqual(parse([crlf.slice(0, i), crlf.slice(i)]), EXPECTED, `cut at ${i}`)
  assert.deepEqual(parse([...crlf]), EXPECTED)
})

test('multiple data lines are joined with a newline', () => {
  assert.deepEqual(parse(['data: line one\ndata: line two\ndata:\ndata: line four\n\n']),
    [{ event: 'message', data: 'line one\nline two\n\nline four' }])
})

test('the event name defaults to "message"; only ONE leading space of a value is dropped; no space is fine too', () => {
  assert.deepEqual(parse(['data: x\n\n']), [{ event: 'message', data: 'x' }])
  assert.deepEqual(parse(['data:x\n\n']), [{ event: 'message', data: 'x' }])
  assert.deepEqual(parse(['data:   three spaces\n\n']), [{ event: 'message', data: '  three spaces' }])
  assert.deepEqual(parse(['event:custom\ndata:{"a":1}\n\n']), [{ event: 'custom', data: '{"a":1}' }])
})

test('values containing colons are kept whole', () => {
  assert.deepEqual(parse(['data: {"label":"Reading 02:10\u201303:10: ok","url":"http://x/y"}\n\n']),
    [{ event: 'message', data: '{"label":"Reading 02:10\u201303:10: ok","url":"http://x/y"}' }])
})

test('id is passed through; retry and unknown fields are ignored; a line without a colon is a field with an empty value', () => {
  assert.deepEqual(parse(['id: 7\nretry: 1000\nfoo: bar\ndata: x\n\n']), [{ event: 'message', data: 'x', id: '7' }])
  assert.deepEqual(parse(['data\n\n']), [{ event: 'message', data: '' }])
})

test('an event without data is not dispatched; blank lines alone do nothing', () => {
  assert.deepEqual(parse(['event: ping\n\n\n\n']), [])
  assert.deepEqual(parse(['\n\n\n']), [])
  assert.deepEqual(parse(['event: ping\n\ndata: real\n\n']), [{ event: 'message', data: 'real' }]) // "ping" must not leak into the next event
})

test('comments can appear anywhere and never end an event', () => {
  assert.deepEqual(parse(['data: a\n: a comment in the middle\ndata: b\n\n']), [{ event: 'message', data: 'a\nb' }])
  assert.deepEqual(parse([':\n:\n\n']), [])
})

test('an event that never got its blank line is dropped at the end of the stream', () => {
  assert.deepEqual(parse(['data: complete\n\ndata: cut off']), [{ event: 'message', data: 'complete' }])
  assert.deepEqual(parse(['data: complete\n\ndata: cut off\n']), [{ event: 'message', data: 'complete' }])
})

test('events are delivered as soon as their blank line arrives, not at the end', () => {
  const events = []
  const p = createSseParser((e) => events.push(e))
  p.push('event: step_start\ndata: 1\n\n')
  assert.equal(events.length, 1)
  p.push('event: step_result\ndata: 2\n')
  assert.equal(events.length, 1) // not complete yet
  p.push('\n')
  assert.deepEqual(events.map((e) => e.event), ['step_start', 'step_result'])
})

test('a trailing "\\r" at the end of the stream still completes its line', () => {
  assert.deepEqual(parse(['data: x\r\r']), [{ event: 'message', data: 'x' }])
})

test('end() resets so a parser can be reused', () => {
  const events = []
  const p = createSseParser((e) => events.push(e))
  p.push('data: half')
  p.end()
  p.push('data: whole\n\n')
  assert.deepEqual(events, [{ event: 'message', data: 'whole' }])
})
