import { excerpt } from './highlight.js'
import { LANGUAGES } from './languages.js'

/** The parts of a search hit that decide what a result card shows. */
export interface SnippetHit {
  text: string
  text_en?: string | null
  language?: string | null
  highlight?: string | null
}

export type SnippetKind = 'original' | 'english'

export interface SnippetPart {
  kind: SnippetKind
  text: string
}

export interface SnippetChoice {
  primary: SnippetPart
  /** The other version, shown smaller and grey below; null when there is only one (English video, or the translation is pending). */
  secondary: SnippetPart | null
}

export interface Segment {
  text: string
  /** True for a matched word (or, as a fallback, the best sentence). */
  hit: boolean
}

const SCRIPTS: [string, RegExp][] = [
  ['latin', /\p{Script=Latin}/u],
  ['devanagari', /\p{Script=Devanagari}/u],
  ['bengali', /\p{Script=Bengali}/u],
  ['gurmukhi', /\p{Script=Gurmukhi}/u],
  ['gujarati', /\p{Script=Gujarati}/u],
  ['oriya', /\p{Script=Oriya}/u],
  ['tamil', /\p{Script=Tamil}/u],
  ['telugu', /\p{Script=Telugu}/u],
  ['kannada', /\p{Script=Kannada}/u],
  ['malayalam', /\p{Script=Malayalam}/u],
  ['arabic', /\p{Script=Arabic}/u],
]

/** The script most of the letters in `text` are written in ("latin", "tamil"...), "other" for anything unlisted, "" if it has no letters. */
export function scriptOf(text: string): string {
  const counts = new Map<string, number>()
  let letters = 0
  for (const ch of text) {
    if (!/\p{L}/u.test(ch)) continue
    letters++
    const name = SCRIPTS.find(([, re]) => re.test(ch))?.[0] ?? 'other'
    counts.set(name, (counts.get(name) ?? 0) + 1)
  }
  if (!letters) return ''
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0]
}

/** True for "en", "en-IN", "en_US"... */
export function isEnglish(language: string | null | undefined): boolean {
  return !!language && /^en([-_]|$)/i.test(language.trim())
}

/**
 * Display name of a video's language for the badge: "ta-IN" (Sarvam) and "ta" (Whisper) both give "Tamil". Null when the language is
 * unknown or unrecognised, so no badge is shown rather than a wrong one.
 */
export function languageLabel(code: string | null | undefined): string | null {
  const c = (code ?? '').trim()
  if (!c) return null
  const exact = LANGUAGES.find((l) => l.code.toLowerCase() === c.toLowerCase())
  if (exact) return exact.label
  const base = c.split(/[-_]/)[0].toLowerCase()
  const byBase = LANGUAGES.find((l) => l.code.split('-')[0] === base)
  if (byBase) return byBase.label
  if (base === 'en') return 'English'
  try {
    const name = new Intl.DisplayNames(['en'], { type: 'language' }).of(base)
    return name && name !== base ? name : null
  } catch {
    return null
  }
}

/**
 * Which text a result shows first.
 *  - English video, no translation yet, or a translation identical to the original: the original only (nothing to compare, no error).
 *  - Query in the same script as the chunk (a Tamil query on Tamil text, Hinglish on a romanised chunk): original first, English below.
 *  - Otherwise (a Latin-script query on a non-Latin chunk, which is the tester's case): English first, original below.
 * `swapped` is the user's "Show original / Show English" toggle: it exchanges the two.
 */
export function chooseSnippet(hit: SnippetHit, query: string, swapped = false): SnippetChoice {
  const original: SnippetPart = { kind: 'original', text: hit.text }
  const en = (hit.text_en ?? '').trim()
  const nonEnglish = hit.language ? !isEnglish(hit.language) : scriptOf(hit.text) !== 'latin'
  if (!nonEnglish || !en || en === hit.text.trim()) return { primary: original, secondary: null }
  const english: SnippetPart = { kind: 'english', text: en }
  const qs = scriptOf(query)
  const originalFirst = qs !== '' && qs === scriptOf(hit.text)
  const [primary, secondary] = originalFirst ? [original, english] : [english, original]
  return swapped ? { primary: secondary, secondary: primary } : { primary, secondary }
}

const STOP = new Set('a an and are as at be by do does for from how i in is it of on or that the this to was what when where which who why with you'.split(' '))
const WORD = /[\p{L}\p{M}\p{N}]+/gu

/** The words of a query worth highlighting: lower-cased, no one-letter words, no English stop words. */
export function queryTerms(query: string): string[] {
  const out = new Set<string>()
  for (const w of query.toLowerCase().match(WORD) ?? []) if (w.length > 1 && !STOP.has(w)) out.add(w)
  return [...out]
}

/** English-only crude stem so "speaking" / "speaks" / "speak" meet. Short words are left alone. */
function stem(w: string): string {
  for (const suffix of ['ing', 'es', 'ed', 's']) if (w.length - suffix.length >= 4 && w.endsWith(suffix)) return w.slice(0, -suffix.length)
  return w
}

/**
 * Splits `text` into plain and matched parts for display, clipped to about `max` characters around the first match. English text
 * matches by stem; original text matches whole words exactly. If no query word occurs in the original text (a Latin query on Tamil)
 * and the backend's best-sentence `highlight` is a substring of it, that sentence is marked instead.
 */
export function markSegments(text: string, terms: string[], opts: { kind: SnippetKind; highlight?: string | null; max?: number }): Segment[] {
  const max = opts.max ?? 280
  const flat = text.replace(/\s+/g, ' ').trim()
  const english = opts.kind === 'english'
  const stems = english ? terms.map(stem) : terms
  const spans: [number, number][] = []
  for (const m of flat.matchAll(WORD)) {
    const w = m[0].toLowerCase()
    const hit = english ? stems.some((s) => w.startsWith(s)) : terms.includes(w)
    if (hit) spans.push([m.index, m.index + m[0].length])
  }
  if (!spans.length) {
    if (opts.kind === 'original') {
      const { before, match, after } = excerpt(flat, opts.highlight, max / 2)
      if (match) return [{ text: before, hit: false }, { text: match, hit: true }, { text: after, hit: false }].filter((s) => s.text)
    }
    return flat ? [{ text: clip(flat, 0, max), hit: false }] : []
  }
  let start = Math.max(0, spans[0][0] - Math.floor(max / 3))
  if (start > 0) start = flat.indexOf(' ', start) + 1 || start  // begin at a word boundary
  const end = Math.min(flat.length, start + max)
  const out: Segment[] = []
  let at = start
  for (const [a, b] of spans) {
    if (a < start || b > end) continue
    if (a > at) out.push({ text: flat.slice(at, a), hit: false })
    out.push({ text: flat.slice(a, b), hit: true })
    at = b
  }
  if (at < end) out.push({ text: flat.slice(at, end), hit: false })
  if (start > 0) out[0] = { ...out[0], text: `…${out[0].text}` }
  if (end < flat.length) out[out.length - 1] = { ...out[out.length - 1], text: out[out.length - 1].text.trimEnd() + '…' }
  return out
}

function clip(s: string, start: number, max: number): string {
  if (s.length - start <= max) return s.slice(start)
  const cut = s.slice(start, start + max)
  const space = cut.lastIndexOf(' ')
  return `${space > 0 ? cut.slice(0, space) : cut}…`
}
