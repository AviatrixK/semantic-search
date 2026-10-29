import type { Chunk } from '../api/types.js'

export interface TranscriptLine {
  /** Stable key: "<chunk idx>.<sentence idx>". */
  id: string
  start: number
  end: number
  text: string
  /** True when the requested language has no text for this part yet: `text` is then the ORIGINAL, shown with a "translation pending" label. */
  pending: boolean
}

/** What the transcript is shown in: the video's own language, the English translation, or a Latin-script transcription. */
export type TranscriptLang = 'original' | 'en' | 'roman'

/**
 * The backend's chunks overlap their neighbours by several seconds, and the overlap is whole segments, so a chunk's text
 * starts with exactly the text its predecessor ends with. Returns `next` without that repeated start (and the length
 * of what was removed). Only the longest match that ends/starts on a word boundary is removed.
 */
export function trimOverlap(prev: string, next: string): { text: string; removed: number } {
  const max = Math.min(prev.length, next.length)
  for (let len = max; len > 0; len--) {
    if (len < next.length && next[len] !== ' ') continue // must end on a word boundary in `next`
    const start = prev.length - len
    if (start > 0 && prev[start - 1] !== ' ') continue // and begin on one in `prev`
    if (prev.endsWith(next.slice(0, len))) return { text: next.slice(len).trimStart(), removed: len }
  }
  return { text: next, removed: 0 }
}

const ABBREVIATIONS = new Set(['mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs', 'e.g', 'i.e', 'u.s', 'u.k', 'fig'])

/** Sentences of a text with the character offset where each starts. Not after abbreviations, only before a capital/digit/quote. */
export function splitSentences(text: string): { text: string; offset: number }[] {
  const out: { text: string; offset: number }[] = []
  const re = /[.!?\u0964]+["')\]]*\s+/g // \u0964 = the Hindi danda
  let start = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text))) {
    const end = m.index + m[0].length
    const next = text[end] ?? ''
    // A sentence starts at a capital, digit or quote, or at a letter of a script without capitals (Tamil, Hindi...): lowercase Latin
    // text after a full stop is not a new sentence ("Hmm... okay").
    const uncased = /\p{L}/u.test(next) && next === next.toUpperCase() && next === next.toLowerCase()
    if (!(/[A-Z0-9"'(\[]/.test(next) || uncased)) continue
    const prevWord = (text.slice(0, m.index).split(/\s+/).pop() ?? '').replace(/^["'(\[]+/, '').toLowerCase()
    if (m[0][0] === '.' && (ABBREVIATIONS.has(prevWord) || (prevWord.length === 1 && /[a-z]/.test(prevWord) && prevWord !== 'i'))) continue
    out.push({ text: text.slice(start, end).trim(), offset: start })
    start = end
  }
  if (text.slice(start).trim()) out.push({ text: text.slice(start).trim(), offset: start })
  return out
}

const squash = (s: string) => s.replace(/\s+/g, ' ').trim()
const sameSentence = (a: string, b: string) => squash(a).toLowerCase() === squash(b).toLowerCase()

/**
 * Drops from the start of a chunk's translated (or romanised) text what the previous chunk already said. The overlap is exact only in the
 * original text: each chunk is translated on its own, so the wording of the repeated part may differ. `fraction` is how much of the
 * ORIGINAL text was the repeat (0..1). Two ways to find the cut, both on sentence starts:
 *  1. the first sentences are the same as the previous chunk's last ones (exact, and within 0.3 of the expected position), or
 *  2. the sentence start nearest to `fraction` of the text (word starts when there is only one sentence, e.g. romanised text without capitals).
 */
export function dropRepeatedStart(text: string, prevText: string | null, fraction: number): string {
  if (fraction <= 0 || !text) return text
  const sentences = splitSentences(text)
  const starts = sentences.length > 1 ? sentences.map((s) => s.offset) : [...text.matchAll(/\S+/g)].map((m) => m.index)
  const target = Math.min(1, fraction) * text.length
  const cutAt = (offset: number) => text.slice(offset).trim()
  if (prevText && sentences.length > 1) {
    const prev = splitSentences(prevText)
    for (let k = Math.min(sentences.length, prev.length); k >= 1; k--) {
      const same = sentences.slice(0, k).every((s, i) => sameSentence(s.text, prev[prev.length - k + i].text))
      if (!same) continue
      const offset = k < sentences.length ? sentences[k].offset : text.length
      if (Math.abs(offset - target) <= 0.3 * text.length) return cutAt(offset)
      break
    }
  }
  let best = text.length // dropping everything is a candidate too (a chunk that only repeats)
  for (const s of [0, ...starts]) if (Math.abs(s - target) < Math.abs(best - target)) best = s
  return cutAt(best)
}

/**
 * Turns overlapping chunks into one continuous, non-repeating transcript of sentence-sized lines.
 * The new (non-repeated) part of a chunk begins where the previous chunk ended; the chunks carry no per-sentence times,
 * so each sentence's start is interpolated by its position in the text (accurate to a second or two).
 *
 * `lang` picks the text: the original, or the English / romanised one. The overlap is always found in the original text (where it is exact)
 * and carried over to the other language with `dropRepeatedStart`. A chunk without text in that language yields its ORIGINAL lines
 * flagged `pending`, never a gap.
 */
export function buildTranscript(chunks: Chunk[], lang: TranscriptLang = 'original'): TranscriptLine[] {
  const sorted = [...chunks].sort((a, b) => a.idx - b.idx)
  const lines: TranscriptLine[] = []
  let prev: Chunk | null = null
  for (const chunk of sorted) {
    const full = squash(chunk.text)
    let text = full
    let begin = chunk.start_sec
    let fraction = 0
    if (prev && chunk.start_sec < prev.end_sec) {
      const overlap = trimOverlap(squash(prev.text), full)
      if (overlap.removed > 0) {
        text = overlap.text
        fraction = full.length ? overlap.removed / full.length : 0
        begin = Math.max(prev.end_sec, chunk.start_sec) // new material starts where the previous chunk stopped
      }
    }
    let pending = false
    if (lang !== 'original') {
      const own = squash((lang === 'en' ? chunk.text_en : chunk.text_roman) ?? '')
      if (own) {
        const before = prev ? squash((lang === 'en' ? prev.text_en : prev.text_roman) ?? '') : ''
        text = dropRepeatedStart(own, before || null, fraction)
      } else {
        pending = full.length > 0
      }
    }
    prev = chunk
    if (!text) continue
    const span = Math.max(0, chunk.end_sec - begin)
    splitSentences(text).forEach((s, j) => {
      const start = begin + (text.length ? (s.offset / text.length) * span : 0)
      lines.push({ id: `${chunk.idx}.${j}`, start, end: chunk.end_sec, text: s.text, pending })
    })
  }
  // each line ends where the next one starts
  for (let i = 0; i < lines.length - 1; i++) lines[i].end = Math.max(lines[i].start, lines[i + 1].start)
  return lines
}

/** The request that makes the player jump to a transcript line (the line's own start, whatever language it is shown in). */
export function seekRequest(videoId: string, line: Pick<TranscriptLine, 'start'>, nonce: number) {
  return { videoId, t: line.start, play: true, nonce }
}

/** Index of the line being spoken at `time` (the last line that started), or -1 before the first line. */
export function activeLineIndex(lines: readonly TranscriptLine[], time: number): number {
  let lo = 0
  let hi = lines.length - 1
  let found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (lines[mid].start <= time) { found = mid; lo = mid + 1 } else hi = mid - 1
  }
  return found
}
