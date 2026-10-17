import test from 'node:test'
import assert from 'node:assert/strict'
import { applyTheme, nextTheme, parseTheme, themeText } from '../.test-build/lib/theme.js'
import { documentTitle, pageName } from '../.test-build/lib/titles.js'

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
  const cases = { '/': 'Home', '/search': 'Search results', '/library': 'Library', '/admin': 'Admin', '/login': 'Log in', '/register': 'Register',
    '/forbidden': 'Not allowed', '/watch/abc': 'Watch', '/watch': 'Watch', '/nope': 'Page not found', '/watchdog': 'Page not found',
    '/ask': 'Page not found', '/ask/': 'Page not found' } // Ask was removed: its address is an ordinary unknown page
  for (const [path, name] of Object.entries(cases)) assert.equal(pageName(path), name, path)
  assert.equal(documentTitle('/library'), 'Library · Semantic Video Search')
})
