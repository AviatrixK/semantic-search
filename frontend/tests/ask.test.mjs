import test from 'node:test'
import assert from 'node:assert/strict'
import { clock, segmentAnswer } from '../.test-build/lib/answerText.js'
import {
  CHAT_STORAGE_KEY, INTERRUPTED, MAX_STORED_MESSAGES, chatReducer, initialChat, isPending, restoreMessages, serializeMessages,
} from '../.test-build/lib/chat.js'
import { ApiError, describeError } from '../.test-build/api/errors.js'

const cite = (n, extra = {}) => ({ n, video_id: `vid-${n}`, title: `Video ${n}`, start_sec: n * 30, end_sec: n * 30 + 30, ...extra })
const run = (...actions) => actions.reduce((s, a) => chatReducer(s, a), initialChat)
const send = (q = 'How do I sound sure?', n = 1) => ({ type: 'send', userId: `u${n}`, assistantId: `a${n}`, question: q })

// ---- answer text -> text + chips
test('segmentAnswer turns known markers into citation segments and keeps the text around them', () => {
  const s = segmentAnswer('Use rhythm [1]. Pause more [2][3].', [cite(1), cite(2), cite(3)])
  assert.deepEqual(s.map((x) => (x.type === 'text' ? x.text : `<${x.citation.n}>`)), ['Use rhythm ', '<1>', '. Pause more ', '<2>', '<3>', '.'])
  assert.equal(s[1].citation.title, 'Video 1')
})

test('segmentAnswer: markers without a matching citation stay plain text', () => {
  const s = segmentAnswer('Known [1], unknown [9], and [sic].', [cite(1)])
  assert.deepEqual(s.map((x) => (x.type === 'text' ? x.text : `<${x.citation.n}>`)), ['Known ', '<1>', ', unknown [9], and [sic].'])
})

test('segmentAnswer: edges and trivial inputs', () => {
  assert.deepEqual(segmentAnswer('', [cite(1)]), [])
  assert.deepEqual(segmentAnswer('No markers here.', [cite(1)]), [{ type: 'text', text: 'No markers here.' }])
  assert.deepEqual(segmentAnswer('[1]', [cite(1)]).map((x) => x.type), ['cite'])
  assert.deepEqual(segmentAnswer('[1] start and end [1]', [cite(1)]).map((x) => x.type), ['cite', 'text', 'cite'])
  assert.deepEqual(segmentAnswer('Claim [1].', []).map((x) => x.type), ['text'])
})

test('segmentAnswer never produces markup: angle brackets are just text', () => {
  const s = segmentAnswer('<img src=x onerror=alert(1)> [1] <script>', [cite(1)])
  assert.ok(s.every((x) => x.type === 'cite' || typeof x.text === 'string'))
  assert.equal(s[0].text, '<img src=x onerror=alert(1)> ')
})

test('clock formats mm:ss for tooltips', () => {
  assert.deepEqual([clock(0), clock(283.9), clock(3912), clock(-5)], ['00:00', '04:43', '65:12', '00:00'])
})

// ---- chat state machine
test('send adds the question and a pending reply', () => {
  const s = run(send())
  assert.deepEqual(s.messages.map((m) => [m.role, m.role === 'user' ? m.text : m.status]), [['user', 'How do I sound sure?'], ['assistant', 'pending']])
  assert.equal(isPending(s), true)
})

test('only one question can be pending at a time', () => {
  const s = run(send('first question', 1))
  assert.equal(chatReducer(s, send('second question', 2)), s) // same object: ignored
})

test('answered fills the reply with text and citations; failed records the reason', () => {
  const done = run(send(), { type: 'answered', id: 'a1', text: 'Pause [1].', citations: [cite(1)] })
  const a = done.messages[1]
  assert.deepEqual([a.status, a.text, a.citations.length, a.error], ['done', 'Pause [1].', 1, null])
  assert.equal(isPending(done), false)
  const failed = run(send(), { type: 'failed', id: 'a1', message: 'The answer service is busy.' })
  assert.deepEqual([failed.messages[1].status, failed.messages[1].error], ['error', 'The answer service is busy.'])
})

test('late or unknown results change nothing', () => {
  const answered = run(send(), { type: 'answered', id: 'a1', text: 'x', citations: [] })
  assert.equal(chatReducer(answered, { type: 'failed', id: 'a1', message: 'late' }), answered)
  assert.equal(chatReducer(answered, { type: 'answered', id: 'nope', text: 'x', citations: [] }), answered)
  assert.equal(chatReducer(initialChat, { type: 'clear' }), initialChat)
})

test('retry turns a failed reply back into a pending one, but not while another is pending', () => {
  const failed = run(send('q one', 1), { type: 'failed', id: 'a1', message: 'busy' })
  const retried = chatReducer(failed, { type: 'retry', id: 'a1' })
  assert.deepEqual([retried.messages[1].status, retried.messages[1].error], ['pending', null])
  const second = run(send('q one', 1), { type: 'failed', id: 'a1', message: 'busy' }, send('q two', 2))
  assert.equal(chatReducer(second, { type: 'retry', id: 'a1' }), second)
  assert.equal(chatReducer(retried, { type: 'retry', id: 'a1' }), retried) // not an error any more
})

test('cancel removes the pending reply and the question that caused it', () => {
  const s = run(send('keep me', 1), { type: 'answered', id: 'a1', text: 'ok', citations: [] }, send('cancel me', 2))
  const c = chatReducer(s, { type: 'cancel', id: 'a2' })
  assert.deepEqual(c.messages.map((m) => m.id), ['u1', 'a1'])
  assert.equal(chatReducer(c, { type: 'cancel', id: 'a1' }), c) // only pending replies can be cancelled
})

test('clear empties the conversation', () => {
  assert.deepEqual(run(send(), { type: 'clear' }).messages, [])
})

// ---- sessionStorage round trip
test('a conversation survives serialize -> restore', () => {
  const s = run(send('q1', 1), { type: 'answered', id: 'a1', text: 'A [1].', citations: [cite(1)] }, send('q2', 2), { type: 'failed', id: 'a2', message: 'busy' })
  assert.deepEqual(restoreMessages(serializeMessages(s.messages)), s.messages)
})

test('a reply that was pending when the page went away comes back as a retryable error', () => {
  const s = run(send('still thinking', 1))
  const back = restoreMessages(serializeMessages(s.messages))
  assert.deepEqual([back[1].status, back[1].error, back[1].question], ['error', INTERRUPTED, 'still thinking'])
  assert.equal(isPending({ messages: back }), false) // the input is not blocked after coming back
})

test('restore ignores garbage instead of trusting it', () => {
  assert.deepEqual(restoreMessages(null), [])
  assert.deepEqual(restoreMessages(''), [])
  assert.deepEqual(restoreMessages('not json'), [])
  assert.deepEqual(restoreMessages('{"a":1}'), [])
  const mixed = JSON.stringify([
    { id: 'u1', role: 'user', text: 'ok' },
    { id: 'x', role: 'admin', text: 'bad role' },
    { role: 'user', text: 'no id' },
    { id: 'u2', role: 'user', text: 5 },
    { id: 'a1', role: 'assistant', status: 'done', question: 'q', text: 'fine', citations: [cite(1), { n: 'x' }, null, 5] },
    { id: 'a2', role: 'assistant', status: 'weird', question: 'q', text: '' },
    'string', 7, null,
  ])
  const restored = restoreMessages(mixed)
  assert.deepEqual(restored.map((m) => m.id), ['u1', 'a1'])
  assert.equal(restored[1].citations.length, 1) // only the well-formed citation
})

test('only the most recent messages are kept', () => {
  const many = Array.from({ length: MAX_STORED_MESSAGES + 10 }, (_, i) => ({ id: `u${i}`, role: 'user', text: `m${i}` }))
  const restored = restoreMessages(serializeMessages(many))
  assert.equal(restored.length, MAX_STORED_MESSAGES)
  assert.equal(restored.at(-1).id, `u${MAX_STORED_MESSAGES + 9}`)
  assert.equal(CHAT_STORAGE_KEY, 'svs.chat.v1')
})

// ---- error text for /api/ask
test('describeError (ask): the backend 503 reason is shown, 429 uses Retry-After, others are generic', () => {
  assert.equal(describeError(new ApiError(503, 'Ask is not set up yet: an administrator needs to configure GEMINI_API_KEY.'), 'ask'),
    'Ask is not set up yet: an administrator needs to configure GEMINI_API_KEY.')
  assert.equal(describeError(new ApiError(429, 'Too many', 20), 'ask'), 'Too many attempts. Please wait 20 seconds and try again.')
  assert.match(describeError(new ApiError(500, 'boom'), 'ask'), /went wrong on the server/)
  assert.equal(describeError(new ApiError(422, 'Question is too short'), 'ask'), 'Question is too short')
  assert.match(describeError(new ApiError(503, 'x')), /went wrong on the server/) // outside "ask" a 503 stays generic
})
