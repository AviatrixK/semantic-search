import test from 'node:test'
import assert from 'node:assert/strict'
import { applyTheme, nextTheme, parseTheme, themeText } from '../.test-build/lib/theme.js'
import { documentTitle, pageName } from '../.test-build/lib/titles.js'
import { copyText } from '../.test-build/lib/clipboard.js'
import { ASK_MODES, parseAskMode } from '../.test-build/lib/askMode.js'
import { answerAsText } from '../.test-build/lib/answerText.js'
import { chatReducer, initialChat } from '../.test-build/lib/chat.js'

// ---- theme
test('parseTheme: only "dark" is dark; everything else (nothing saved, an old "system", junk) is light', () => {
  assert.equal(parseTheme('dark'), 'dark')
  for (const v of ['light', 'system', null, undefined, '', 'Dark', 'blue', 5, {}, '<script>']) assert.equal(parseTheme(v), 'light', String(v))
})

test('nextTheme switches between light and dark', () => {
  assert.deepEqual(['light', 'dark'].map(nextTheme), ['dark', 'light'])
})

test('applyTheme always sets data-theme, so the CSS never depends on the OS setting', () => {
  const attrs = new Map()
  const root = { setAttribute: (k, v) => attrs.set(k, v), removeAttribute: (k) => attrs.delete(k) }
  applyTheme('dark', root)
  assert.equal(attrs.get('data-theme'), 'dark')
  applyTheme('light', root)
  assert.equal(attrs.get('data-theme'), 'light')
})

test('themeText names the current theme and what a click switches to', () => {
  assert.deepEqual(themeText('light'), { current: 'Light', next: 'Dark', label: 'Theme: Light. Switch to dark.' })
  assert.equal(themeText('dark').label, 'Theme: Dark. Switch to light.')
})

// ---- page titles
test('pageName / documentTitle cover every route, with or without a trailing slash', () => {
  const cases = { '/': 'Home', '/search': 'Search results', '/ask': 'Ask', '/ask/': 'Ask', '/library': 'Library', '/admin': 'Admin', '/login': 'Log in', '/register': 'Register',
    '/forbidden': 'Not allowed', '/watch/abc': 'Watch', '/watch': 'Watch', '/nope': 'Page not found', '/watchdog': 'Page not found' }
  for (const [path, name] of Object.entries(cases)) assert.equal(pageName(path), name, path)
  assert.equal(documentTitle('/ask'), 'Ask · Semantic Video Search')
})

// ---- clipboard
test('copyText reports success, refusal, and a missing clipboard API', async () => {
  const written = []
  assert.equal(await copyText('hello', { writeText: async (t) => { written.push(t) } }), true)
  assert.deepEqual(written, ['hello'])
  assert.equal(await copyText('x', { writeText: async () => { throw new Error('denied') } }), false)
  assert.equal(await copyText('x', {}), false)
  assert.equal(await copyText('x', undefined), false)
})

// ---- ask mode
test('parseAskMode accepts only the modes the server accepts', () => {
  assert.deepEqual([...ASK_MODES], ['auto', 'rag', 'agent'])
  for (const ok of ASK_MODES) assert.equal(parseAskMode(ok), ok)
  for (const bad of [null, 'search', 'AGENT', '', 7]) assert.equal(parseAskMode(bad), 'auto')
})

// ---- copying an answer
const cite = (n, title, start) => ({ n, video_id: `v${n}`, title, start_sec: start, end_sec: start + 30 })

test('answerAsText keeps the [n] markers and lists the sources with their times', () => {
  assert.equal(answerAsText('  Pause often [1] and breathe [2].  ', [cite(1, 'How to Speak', 283), cite(2, 'TED talk', 65)]),
    'Pause often [1] and breathe [2].\n\nSources:\n[1] How to Speak @ 04:43\n[2] TED talk @ 01:05')
  assert.equal(answerAsText('No sources here.', []), 'No sources here.')
})

// ---- regenerate
const base = { id: 'a1', role: 'assistant', status: 'done', question: 'q', text: 'old', citations: [cite(1, 'T', 1)], error: null, route: 'agent', routeReason: 'r',
  trace: [{ step: 1, index: 0, tool: 't', label: 'l', args: {}, summary: 's', latencyMs: 1, status: 'done' }], usage: { llmCalls: 2, tokens: 9 } }

test('regenerate turns a finished answer back into a clean pending one', () => {
  const s = chatReducer({ messages: [{ id: 'u1', role: 'user', text: 'q' }, base] }, { type: 'regenerate', id: 'a1' })
  const a = s.messages[1]
  assert.deepEqual([a.status, a.text, a.citations, a.route, a.trace, a.usage, a.question], ['pending', '', [], null, [], null, 'q'])
})

test('regenerate is ignored while another answer is pending, for errors, and for unknown ids', () => {
  const pending = { ...base, id: 'a2', status: 'pending' }
  const busy = { messages: [base, pending] }
  assert.equal(chatReducer(busy, { type: 'regenerate', id: 'a1' }), busy)
  const failed = { messages: [{ ...base, status: 'error', error: 'x' }] }
  assert.equal(chatReducer(failed, { type: 'regenerate', id: 'a1' }), failed)
  assert.equal(chatReducer(initialChat, { type: 'regenerate', id: 'nope' }), initialChat)
})
