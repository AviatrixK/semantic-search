export interface Excerpt {
  before: string
  /** The sentence to show in bold ("" if the backend sent no highlight or it is not found in the text). */
  match: string
  after: string
}

const squash = (s: string) => s.replace(/\s+/g, ' ').trim()

/** Keeps at most `max` characters from the END of `s`, starting at a word boundary, with a leading ellipsis. */
function clipStart(s: string, max: number): string {
  if (s.length <= max) return s
  const cut = s.slice(s.length - max)
  const space = cut.indexOf(' ')
  return `…${space >= 0 ? cut.slice(space + 1) : cut}`
}

/** Keeps at most `max` characters from the START of `s`, ending at a word boundary, with a trailing ellipsis. */
function clipEnd(s: string, max: number): string {
  if (s.length <= max) return s
  const cut = s.slice(0, max)
  const space = cut.lastIndexOf(' ')
  return `${space >= 0 ? cut.slice(0, space) : cut}…`
}

/**
 * Splits a chunk's text around the best-matching sentence so a result card can show it in bold with a little context.
 * Long text is clipped to about `maxSide` characters on each side of the match.
 */
export function excerpt(text: string, highlight: string | null | undefined, maxSide = 140): Excerpt {
  const t = squash(text)
  const h = highlight ? squash(highlight) : ''
  const i = h ? t.indexOf(h) : -1
  if (i < 0) return { before: '', match: '', after: clipEnd(t, maxSide * 2) }
  return {
    before: clipStart(t.slice(0, i), maxSide),
    match: h,
    after: clipEnd(t.slice(i + h.length), maxSide),
  }
}
