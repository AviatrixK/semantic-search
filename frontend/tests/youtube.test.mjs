import test from 'node:test'
import assert from 'node:assert/strict'
import { formatClock, formatRelative } from '../.test-build/lib/format.js'
import { SORT_OPTIONS, sortVideos } from '../.test-build/lib/videos.js'
import { DEFAULT_FILTERS, searchPagePath } from '../.test-build/lib/searchParams.js'

test('formatClock: YouTube-style lengths, hours only when needed, empty when unknown', () => {
  assert.deepEqual([0, 5, 59, 60, 283.9, 1668, 3599, 3600, 3723, 36000].map(formatClock), ['0:00', '0:05', '0:59', '1:00', '4:43', '27:48', '59:59', '1:00:00', '1:02:03', '10:00:00'])
  for (const bad of [null, undefined, -1, NaN, Infinity]) assert.equal(formatClock(bad), '', String(bad))
})

test('formatRelative: picks the largest unit, pluralises, and never says "in the future"', () => {
  const now = Date.parse('2026-10-10T12:00:00Z')
  const ago = (s) => new Date(now - s * 1000).toISOString()
  assert.equal(formatRelative(ago(5), now), 'just now')
  assert.equal(formatRelative(ago(60), now), '1 minute ago')
  assert.equal(formatRelative(ago(59 * 60), now), '59 minutes ago')
  assert.equal(formatRelative(ago(3600), now), '1 hour ago')
  assert.equal(formatRelative(ago(86400 * 3), now), '3 days ago')
  assert.equal(formatRelative(ago(86400 * 7), now), '1 week ago')
  assert.equal(formatRelative(ago(86400 * 45), now), '1 month ago')
  assert.equal(formatRelative(ago(86400 * 800), now), '2 years ago')
  assert.equal(formatRelative(new Date(now + 99999).toISOString(), now), 'just now') // clock skew
  assert.equal(formatRelative('not a date', now), '')
})

const v = (id, created, dur) => ({ id, created_at: created, duration_sec: dur })

test('sortVideos: newest, oldest, longest and shortest; unknown durations last; stable; input untouched', () => {
  const list = [v('a', '2026-01-02T00:00:00Z', 100), v('b', '2026-03-01T00:00:00Z', null), v('c', '2026-02-01T00:00:00Z', 300), v('d', '2026-02-01T00:00:00Z', 100)]
  const ids = (sort) => sortVideos(list, sort).map((x) => x.id).join('')
  assert.equal(ids('newest'), 'bcda') // c and d tie on date: incoming order kept
  assert.equal(ids('oldest'), 'acdb')
  assert.equal(ids('longest'), 'cadb') // a and d tie on 100: incoming order kept; b (unknown) last
  assert.equal(ids('shortest'), 'adcb')
  assert.deepEqual(list.map((x) => x.id), ['a', 'b', 'c', 'd'])
  assert.deepEqual(SORT_OPTIONS.map((o) => o.value), ['newest', 'oldest', 'longest', 'shortest'])
  assert.deepEqual(sortVideos([], 'longest'), [])
})

test('sortVideos: an unreadable date counts as the oldest', () => {
  assert.equal(sortVideos([v('x', 'junk', 1), v('y', '2026-01-01T00:00:00Z', 1)], 'newest').map((q) => q.id).join(''), 'yx')
})

test('searchPagePath builds the results address, keeping filters and dropping empty queries', () => {
  assert.equal(searchPagePath('how to pause'), '/search?q=how+to+pause')
  assert.equal(searchPagePath('  x  ', DEFAULT_FILTERS), '/search?q=x')
  assert.equal(searchPagePath('a&b'), '/search?q=a%26b')
  assert.equal(searchPagePath('q', { mode: 'keyword', videoId: 'dfc14e3c-3b41-45e3-a078-74478f32fecb', after: '2026-01-02' }),
    '/search?q=q&mode=keyword&video=dfc14e3c-3b41-45e3-a078-74478f32fecb&after=2026-01-02')
  assert.equal(searchPagePath('   '), '/search')
})

test('hueFor is stable, in range and differs between ids', async () => {
  const { hueFor } = await import('../.test-build/lib/videos.js')
  const a = hueFor('8f27bfee-4387-448e-bc0f-29cb05d75f80')
  assert.equal(a, hueFor('8f27bfee-4387-448e-bc0f-29cb05d75f80'))
  assert.ok(a >= 0 && a < 360)
  assert.notEqual(a, hueFor('dfc14e3c-3b41-45e3-a078-74478f32fecb'))
  assert.equal(hueFor(''), 0)
})
