import type { Citation } from '../api/types.js'

export type AnswerSegment =
  | { type: 'text'; text: string }
  | { type: 'cite'; citation: Citation }

const MARKER = /\[(\d+)\]/g

/**
 * Splits an answer into text and citation markers. Only a [n] that is in `citations` becomes a clickable chip: the backend
 * already removed numbers that do not exist, and anything else that looks like a marker stays plain text. The answer is
 * rendered as text nodes only (never as HTML), so nothing the model writes can inject markup.
 */
export function segmentAnswer(text: string, citations: readonly Citation[]): AnswerSegment[] {
  const byN = new Map(citations.map((c) => [c.n, c]))
  const out: AnswerSegment[] = []
  let last = 0
  let m: RegExpExecArray | null
  const push = (t: string) => {
    if (!t) return
    const prev = out.at(-1)
    if (prev?.type === 'text') prev.text += t
    else out.push({ type: 'text', text: t })
  }
  MARKER.lastIndex = 0
  while ((m = MARKER.exec(text))) {
    const citation = byN.get(Number(m[1]))
    if (!citation) continue // leave it inside the surrounding text
    push(text.slice(last, m.index))
    out.push({ type: 'cite', citation })
    last = m.index + m[0].length
  }
  push(text.slice(last))
  return out
}

/** "04:43" for a chip's tooltip. */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}
