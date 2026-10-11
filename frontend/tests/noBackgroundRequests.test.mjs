// Source-level guards for the "no background requests" rules, so a later change cannot quietly bring them back.
import test from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { join, relative } from 'node:path'

const src = fileURLToPath(new URL('../src/', import.meta.url))

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name)
    return statSync(p).isDirectory() ? files(p) : /\.(ts|tsx)$/.test(name) ? [p] : []
  })
}
const all = files(src).map((p) => ({ name: relative(src, p).replaceAll('\\', '/'), text: readFileSync(p, 'utf8') }))
// Comments may talk about what is forbidden; only code counts.
const code = (t) => t.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1')

test('no setInterval anywhere in the frontend', () => {
  const offenders = all.filter((f) => /\bsetInterval\s*\(/.test(code(f.text))).map((f) => f.name)
  assert.deepEqual(offenders, [])
})

test('the old polling helpers are gone', () => {
  assert.deepEqual(all.filter((f) => /startPolling|useJobPolling/.test(code(f.text))).map((f) => f.name), [])
})

test('every <video> has preload="none"', () => {
  const tags = all.flatMap((f) => [...code(f.text).matchAll(/<video\b[^>]*>/g)].map((m) => ({ file: f.name, tag: m[0] })))
  assert.ok(tags.length >= 1, 'expected the player to exist')
  assert.deepEqual(tags.filter((t) => !/preload="none"/.test(t.tag)).map((t) => t.file), [])
})

test('there is exactly one <video> element in the app: the player', () => {
  const files = all.filter((f) => /<video\b/.test(code(f.text))).map((f) => f.name)
  assert.deepEqual(files, ['components/VideoPlayer.tsx'])
})

test('only the player asks for stream URLs (results and cards never do)', () => {
  const users = all.filter((f) => /streamUrls\./.test(code(f.text))).map((f) => f.name).sort()
  assert.deepEqual(users, ['components/VideoPlayer.tsx', 'context/AuthContext.tsx']) // AuthContext only clears the cache on logout
})
