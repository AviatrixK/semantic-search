// Regenerates public/favicon.ico (a 32x32 PNG inside an ICO container) from the same graduation-cap shape as favicon.svg.
// No dependencies: Node's zlib only. Run: node scripts/make-favicon.mjs
import { deflateSync, crc32 } from 'node:zlib'
import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const SIZE = 32
const SS = 4 // samples per pixel side, for smooth edges
const BRAND = [0x4f, 0x46, 0xe5]
const WHITE = [255, 255, 255]

// Shapes in the 24-unit space of the icon, shifted by 4 units so the cap sits centred in the 32-unit tile.
const inPoly = (x, y, pts) => {
  let inside = false
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}
const BOARD = [[2.5, 9.5], [12, 5], [21.5, 9.5], [12, 14]] // the flat top
const BAND = [[6, 11.8], [12, 14.2], [18, 11.8], [18, 16], [12, 18.2], [6, 16]] // the cap body under it
const TASSEL = [[20.8, 9.6], [22.2, 9.6], [22.2, 15.2], [20.8, 15.2]]

function inRoundedSquare(x, y, r) {
  const cx = Math.min(Math.max(x, r), SIZE - r)
  const cy = Math.min(Math.max(y, r), SIZE - r)
  return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
}

const raw = Buffer.alloc((SIZE * 4 + 1) * SIZE)
for (let py = 0; py < SIZE; py++) {
  raw[py * (SIZE * 4 + 1)] = 0 // PNG filter: none
  for (let px = 0; px < SIZE; px++) {
    let r = 0, g = 0, b = 0, a = 0
    for (let sy = 0; sy < SS; sy++) {
      for (let sx = 0; sx < SS; sx++) {
        const x = px + (sx + 0.5) / SS, y = py + (sy + 0.5) / SS
        if (!inRoundedSquare(x, y, 7)) continue
        const u = x - 4, v = y - 4
        const c = inPoly(u, v, BOARD) || inPoly(u, v, BAND) || inPoly(u, v, TASSEL) ? WHITE : BRAND
        r += c[0]; g += c[1]; b += c[2]; a += 255
      }
    }
    const n = SS * SS, o = py * (SIZE * 4 + 1) + 1 + px * 4
    const covered = a / 255 || 1
    raw[o] = Math.round(r / covered); raw[o + 1] = Math.round(g / covered); raw[o + 2] = Math.round(b / covered)
    raw[o + 3] = Math.round(a / n)
  }
}

function chunk(type, data) {
  const head = Buffer.alloc(4)
  head.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const tail = Buffer.alloc(4)
  tail.writeUInt32BE(crc32(body) >>> 0)
  return Buffer.concat([head, body, tail])
}
const ihdr = Buffer.alloc(13)
ihdr.writeUInt32BE(SIZE, 0); ihdr.writeUInt32BE(SIZE, 4)
ihdr[8] = 8; ihdr[9] = 6 // 8-bit RGBA
const png = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0)),
])

const dir = Buffer.alloc(22)
dir.writeUInt16LE(0, 0); dir.writeUInt16LE(1, 2); dir.writeUInt16LE(1, 4) // ICONDIR: reserved, type icon, 1 image
dir[6] = SIZE; dir[7] = SIZE; dir[8] = 0; dir[9] = 0 // width, height, no palette, reserved
dir.writeUInt16LE(1, 10); dir.writeUInt16LE(32, 12) // planes, bits per pixel
dir.writeUInt32LE(png.length, 14); dir.writeUInt32LE(22, 18) // size and offset of the PNG data
const out = fileURLToPath(new URL('../public/favicon.ico', import.meta.url))
writeFileSync(out, Buffer.concat([dir, png]))
console.log(`wrote ${out} (${22 + png.length} bytes)`)
