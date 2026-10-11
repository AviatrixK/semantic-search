import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const root = new URL('../', import.meta.url)
const file = (p) => fileURLToPath(new URL(p, root))
const html = readFileSync(file('index.html'), 'utf8')

test('index.html declares icons that exist in public/ (Vite serves and copies public/ to the site root)', () => {
  const hrefs = [...html.matchAll(/<link[^>]*rel="icon"[^>]*href="([^"]+)"/g)].map((m) => m[1])
  assert.deepEqual(hrefs.sort(), ['/favicon.ico', '/favicon.svg'])
  for (const href of hrefs) assert.ok(existsSync(file('public' + href)), `public${href} is missing`)
})

test('favicon.ico is a valid ICO holding one 32x32 PNG', () => {
  const b = readFileSync(file('public/favicon.ico'))
  assert.equal(b.readUInt16LE(0), 0) // reserved
  assert.equal(b.readUInt16LE(2), 1) // type: icon
  assert.equal(b.readUInt16LE(4), 1) // one image
  assert.equal(b[6], 32)
  assert.equal(b[7], 32)
  const size = b.readUInt32LE(14)
  const offset = b.readUInt32LE(18)
  assert.equal(offset + size, b.length) // the image fills the rest of the file exactly
  assert.deepEqual([...b.subarray(offset, offset + 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]) // PNG signature
  assert.equal(b.readUInt32BE(offset + 16), 32) // IHDR width
  assert.equal(b.readUInt32BE(offset + 20), 32) // IHDR height
})

test('favicon.svg is a standalone SVG in the brand colour', () => {
  const svg = readFileSync(file('public/favicon.svg'), 'utf8')
  assert.match(svg, /<svg[^>]*xmlns="http:\/\/www\.w3\.org\/2000\/svg"/)
  assert.match(svg, /viewBox="0 0 32 32"/)
  assert.match(svg, /#4f46e5/i) // --brand in index.css
})
