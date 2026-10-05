import test from 'node:test'
import assert from 'node:assert/strict'
import { clock, segmentAnswer } from '../.test-build/lib/answerText.js'
import {
  CHAT_STORAGE_KEY, INTERRUPTED, MAX_STORED_MESSAGES, chatReducer, initialChat, isPending, isValidSessionId, newSessionId, restoreMessages,
  serializeMessages,
} from '../.test-build/lib/chat.js'
import { ApiError, describeError } from '../.test-build/api/errors.js'

const cite = (n, extra = {}) => ({ n, video_id: `vid-${n}`, title: `Video ${n}`, start_sec: n * 30, end_sec: n * 30 + 30, ...extra })
const run = (...actions) => actions.reduce((s, a) => chatReducer(s, a), initialChat)
const send = (q = 'How do I sound sure?', n = 1) => ({ type: 'send', userId: `u${n}`, assistantId: `a${n}`, question: q })
const answered = (id, text, citations = [], extra = {}) => ({
  type: 'answered', id, text, citations, route: 'rag', routeReason: 'single question', trace: [], usage: { llmCalls: 1, tokens: 100 }, ...extra,
})

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
  const done = run(send(), answered('a1', 'Pause [1].', [cite(1)]))
  const a = done.messages[1]
  assert.deepEqual([a.status, a.text, a.citations.length, a.error], ['done', 'Pause [1].', 1, null])
  assert.equal(isPending(done), false)
  const failed = run(send(), { type: 'failed', id: 'a1', message: 'The answer service is busy.' })
  assert.deepEqual([failed.messages[1].status, failed.messages[1].error], ['error', 'The answer service is busy.'])
})

test('late or unknown results change nothing', () => {
  const done = run(send(), answered('a1', 'x'))
  assert.equal(chatReducer(done, { type: 'failed', id: 'a1', message: 'late' }), done)
  assert.equal(chatReducer(done, answered('nope', 'x')), done)
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
  const s = run(send('keep me', 1), answered('a1', 'ok'), send('cancel me', 2))
  const c = chatReducer(s, { type: 'cancel', id: 'a2' })
  assert.deepEqual(c.messages.map((m) => m.id), ['u1', 'a1'])
  assert.equal(chatReducer(c, { type: 'cancel', id: 'a1' }), c) // only pending replies can be cancelled
})

test('clear empties the conversation', () => {
  assert.deepEqual(run(send(), { type: 'clear' }).messages, [])
})

// ---- sessionStorage round trip
test('a conversation survives serialize -> restore', () => {
  const s = run(send('q1', 1), answered('a1', 'A [1].', [cite(1)]), send('q2', 2), { type: 'failed', id: 'a2', message: 'busy' })
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

// ---- live progress from the stream
const progress = (id, event) => ({ type: 'progress', id, event })
const start = (step, index, label = 'Searching “x”') => ({ type: 'step_start', step, index, tool: 'search_transcripts', label, args: { query: 'x' } })
const result = (step, index, summary = 'Found 5 clips', error = false) => ({ type: 'step_result', step, index, tool: 'search_transcripts', summary, latency_ms: 120, error })

test('route and step events build up the trace of a pending reply', () => {
  const s = run(send(), progress('a1', { type: 'route', route: 'agent', requested: 'auto', reason: 'comparison' }), progress('a1', start(1, 0)),
    progress('a1', start(1, 1, 'Listing videos')))
  const a = s.messages[1]
  assert.deepEqual([a.route, a.routeReason], ['agent', 'comparison'])
  assert.deepEqual(a.trace.map((t) => [t.step, t.index, t.status, t.summary]), [[1, 0, 'running', null], [1, 1, 'running', null]])
  assert.equal(a.trace[0].label, 'Searching “x”')
  assert.deepEqual(a.trace[0].args, { query: 'x' })
})

test('a step result completes its step (success or error) with summary and latency', () => {
  const s = run(send(), progress('a1', start(1, 0)), progress('a1', start(1, 1)), progress('a1', result(1, 1, 'Error: bad', true)), progress('a1', result(1, 0)))
  assert.deepEqual(s.messages[1].trace.map((t) => [t.status, t.summary, t.latencyMs]), [['done', 'Found 5 clips', 120], ['error', 'Error: bad', 120]])
})

test('a result for a step whose start was missed is added, and a repeated start is not duplicated', () => {
  const s = run(send(), progress('a1', result(2, 0, 'Read 02:10–03:10')), progress('a1', start(3, 0)), progress('a1', start(3, 0)))
  assert.deepEqual(s.messages[1].trace.map((t) => [t.step, t.status]), [[2, 'done'], [3, 'running']])
})

test('progress is ignored unless the reply is pending (late events after cancel, error or answer)', () => {
  const done = run(send(), answered('a1', 'x'))
  assert.equal(chatReducer(done, progress('a1', start(1, 0))), done)
  assert.equal(chatReducer(run(send()), progress('nope', start(1, 0))).messages[1].trace.length, 0)
  const failed = run(send(), { type: 'failed', id: 'a1', message: 'bad' })
  assert.equal(chatReducer(failed, progress('a1', start(1, 0))), failed)
})

test('the final answer carries the route, the complete trace and usage', () => {
  const trace = [{ step: 1, index: 0, tool: 't', label: 'l', args: {}, summary: 'Found 1 clip', latencyMs: 10, status: 'done' }]
  const s = run(send(), progress('a1', start(1, 0)), answered('a1', 'Done [1].', [cite(1)], { route: 'agent', routeReason: 'comparison', trace, usage: { llmCalls: 3, tokens: 420 } }))
  const a = s.messages[1]
  assert.deepEqual([a.status, a.route, a.routeReason, a.trace, a.usage], ['done', 'agent', 'comparison', trace, { llmCalls: 3, tokens: 420 }])
})

test('retry clears the old route and trace (the failed reply keeps them so the user sees how far it got)', () => {
  const failed = run(send(), progress('a1', { type: 'route', route: 'agent', requested: 'auto', reason: 'r' }), progress('a1', start(1, 0)), { type: 'failed', id: 'a1', message: 'busy' })
  assert.equal(failed.messages[1].trace.length, 1)
  const retried = chatReducer(failed, { type: 'retry', id: 'a1' })
  assert.deepEqual([retried.messages[1].status, retried.messages[1].route, retried.messages[1].trace], ['pending', null, []])
})

test('route and trace survive serialize -> restore', () => {
  const trace = [{ step: 1, index: 0, tool: 't', label: 'l', args: { query: 'x' }, summary: 'Found', latencyMs: 5, status: 'done' }]
  const s = run(send(), answered('a1', 'A [1].', [cite(1)], { route: 'agent', routeReason: 'comparison', trace, usage: { llmCalls: 2, tokens: 9 } }))
  assert.deepEqual(restoreMessages(serializeMessages(s.messages)), s.messages)
})

test('steps that were still running when the page went away come back as interrupted', () => {
  const s = run(send(), progress('a1', { type: 'route', route: 'agent', requested: 'auto', reason: 'r' }), progress('a1', start(1, 0)), progress('a1', start(1, 1)), progress('a1', result(1, 1)))
  const back = restoreMessages(serializeMessages(s.messages))[1]
  assert.deepEqual([back.status, back.error, back.route], ['error', INTERRUPTED, 'agent'])
  assert.deepEqual(back.trace.map((t) => [t.status, t.summary]), [['error', 'Interrupted'], ['done', 'Found 5 clips']])
})

test('old stored conversations without route/trace/usage still load, with defaults', () => {
  const old = JSON.stringify([{ id: 'a1', role: 'assistant', status: 'done', question: 'q', text: 'fine', citations: [] }])
  const [m] = restoreMessages(old)
  assert.deepEqual([m.route, m.routeReason, m.trace, m.usage], [null, null, [], null])
})

test('malformed trace steps, routes and usage in storage are dropped', () => {
  const bad = JSON.stringify([{ id: 'a1', role: 'assistant', status: 'done', question: 'q', text: 't', citations: [], route: 'magic',
    trace: [{ step: 1, index: 0, tool: 't', label: 'ok', status: 'done' }, { step: 'x' }, null, 5], usage: { llmCalls: 'many' } }])
  const [m] = restoreMessages(bad)
  assert.equal(m.route, null)
  assert.equal(m.trace.length, 1)
  assert.equal(m.usage, null)
})

// ---- session id for server-side memory
test('newSessionId makes ids the server accepts', () => {
  const id = newSessionId()
  assert.equal(id.length, 24)
  assert.ok(isValidSessionId(id))
  assert.notEqual(newSessionId(), newSessionId())
  assert.equal(newSessionId(() => 0, 8), 'AAAAAAAA')
  assert.equal(newSessionId(() => 0.9999999, 8), '99999999')
})

test('isValidSessionId matches the backend pattern', () => {
  for (const ok of ['abcdefgh', 'a-b_c-D1234567', 'x'.repeat(64)]) assert.equal(isValidSessionId(ok), true, ok)
  for (const bad of ['short', 'has space here', 'x'.repeat(65), 'semi;colon-1234', '', null, undefined, 12345678]) assert.equal(isValidSessionId(bad), false, String(bad))
})
