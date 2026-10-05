import test from 'node:test'
import assert from 'node:assert/strict'
import { formatArgs, formatLatency, fromServerTrace, groupByStep, routeLabel, stepLine } from '../.test-build/lib/trace.js'

const step = (over = {}) => ({ step: 1, index: 0, tool: 'search_transcripts', label: 'Searching “x”', args: { query: 'x' }, summary: null, latencyMs: null, status: 'running', ...over })

test('stepLine: the label (with an ellipsis) while running, the outcome when done', () => {
  assert.equal(stepLine(step()), 'Searching “x”…')
  assert.equal(stepLine(step({ status: 'done', summary: 'Found 5 clips' })), 'Found 5 clips')
  assert.equal(stepLine(step({ status: 'error', summary: 'Error: Invalid arguments' })), 'Error: Invalid arguments')
  assert.equal(stepLine(step({ status: 'done', summary: null })), 'Searching “x”')
})

test('formatLatency', () => {
  assert.deepEqual([formatLatency(0), formatLatency(87.4), formatLatency(999), formatLatency(1000), formatLatency(1234), formatLatency(12500)],
    ['0 ms', '87 ms', '999 ms', '1.0 s', '1.2 s', '12.5 s'])
  assert.deepEqual([formatLatency(null), formatLatency(undefined), formatLatency(NaN)], ['', '', ''])
})

test('routeLabel names each route and explains it', () => {
  assert.equal(routeLabel('search').label, 'Search results')
  assert.equal(routeLabel('rag').label, 'Direct answer')
  assert.equal(routeLabel('agent').label, 'Research agent')
  assert.match(routeLabel('search').hint, /no AI model/)
  assert.equal(routeLabel(null), null)
})

test('groupByStep puts the parallel calls of one round together and keeps order', () => {
  const steps = [step({ step: 1, index: 0 }), step({ step: 1, index: 1 }), step({ step: 2, index: 0 }), step({ step: 3, index: 0 }), step({ step: 3, index: 1 })]
  assert.deepEqual(groupByStep(steps).map((g) => g.map((s) => `${s.step}.${s.index}`)), [['1.0', '1.1'], ['2.0'], ['3.0', '3.1']])
  assert.deepEqual(groupByStep([]), [])
})

test('formatArgs prints one argument per line, quoting strings', () => {
  assert.equal(formatArgs({ query: 'pacing', k: 5, video_id: null }), 'query: "pacing"\nk: 5\nvideo_id: null')
  assert.equal(formatArgs({}), '(no arguments)')
})

test('fromServerTrace converts the finished trace of an answer', () => {
  const out = fromServerTrace([
    { step: 1, index: 0, tool: 'search_transcripts', args: { query: 'x' }, label: 'Searching', summary: 'Found 2 clips', latency_ms: 120, error: false },
    { step: 2, index: 0, tool: 'list_videos', args: {}, label: 'Listing videos', summary: 'Error: nope', latency_ms: 5, error: true },
  ])
  assert.deepEqual(out.map((s) => [s.status, s.summary, s.latencyMs]), [['done', 'Found 2 clips', 120], ['error', 'Error: nope', 5]])
  assert.deepEqual(fromServerTrace([]), [])
})
