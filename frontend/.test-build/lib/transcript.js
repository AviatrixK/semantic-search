/**
 * The backend's chunks overlap their neighbours by several seconds, and the overlap is whole segments, so a chunk's text
 * starts with exactly the text its predecessor ends with. Returns `next` without that repeated start (and the length
 * of what was removed). Only the longest match that ends/starts on a word boundary is removed.
 */
export function trimOverlap(prev, next) {
    const max = Math.min(prev.length, next.length);
    for (let len = max; len > 0; len--) {
        if (len < next.length && next[len] !== ' ')
            continue; // must end on a word boundary in `next`
        const start = prev.length - len;
        if (start > 0 && prev[start - 1] !== ' ')
            continue; // and begin on one in `prev`
        if (prev.endsWith(next.slice(0, len)))
            return { text: next.slice(len).trimStart(), removed: len };
    }
    return { text: next, removed: 0 };
}
const ABBREVIATIONS = new Set(['mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs', 'e.g', 'i.e', 'u.s', 'u.k', 'fig']);
/** Sentences of a text with the character offset where each starts. Not after abbreviations, only before a capital/digit/quote. */
export function splitSentences(text) {
    const out = [];
    const re = /[.!?]+["')\]]*\s+/g;
    let start = 0;
    let m;
    while ((m = re.exec(text))) {
        const end = m.index + m[0].length;
        const next = text[end] ?? '';
        if (!(/[A-Z0-9"'(\[]/.test(next)))
            continue;
        const prevWord = (text.slice(0, m.index).split(/\s+/).pop() ?? '').replace(/^["'(\[]+/, '').toLowerCase();
        if (m[0][0] === '.' && (ABBREVIATIONS.has(prevWord) || (prevWord.length === 1 && /[a-z]/.test(prevWord) && prevWord !== 'i')))
            continue;
        out.push({ text: text.slice(start, end).trim(), offset: start });
        start = end;
    }
    if (text.slice(start).trim())
        out.push({ text: text.slice(start).trim(), offset: start });
    return out;
}
/**
 * Turns overlapping chunks into one continuous, non-repeating transcript of sentence-sized lines.
 * The new (non-repeated) part of a chunk begins where the previous chunk ended; the chunks carry no per-sentence times,
 * so each sentence's start is interpolated by its position in the text (accurate to a second or two).
 */
export function buildTranscript(chunks) {
    const sorted = [...chunks].sort((a, b) => a.idx - b.idx);
    const lines = [];
    let prev = null;
    for (const chunk of sorted) {
        const full = chunk.text.replace(/\s+/g, ' ').trim();
        let text = full;
        let begin = chunk.start_sec;
        if (prev && chunk.start_sec < prev.end_sec) {
            const overlap = trimOverlap(prev.text.replace(/\s+/g, ' ').trim(), full);
            if (overlap.removed > 0) {
                text = overlap.text;
                begin = Math.max(prev.end_sec, chunk.start_sec); // new material starts where the previous chunk stopped
            }
        }
        prev = chunk;
        if (!text)
            continue;
        const span = Math.max(0, chunk.end_sec - begin);
        splitSentences(text).forEach((s, j) => {
            const start = begin + (text.length ? (s.offset / text.length) * span : 0);
            lines.push({ id: `${chunk.idx}.${j}`, start, end: chunk.end_sec, text: s.text });
        });
    }
    // each line ends where the next one starts
    for (let i = 0; i < lines.length - 1; i++)
        lines[i].end = Math.max(lines[i].start, lines[i + 1].start);
    return lines;
}
/** Index of the line being spoken at `time` (the last line that started), or -1 before the first line. */
export function activeLineIndex(lines, time) {
    let lo = 0;
    let hi = lines.length - 1;
    let found = -1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (lines[mid].start <= time) {
            found = mid;
            lo = mid + 1;
        }
        else
            hi = mid - 1;
    }
    return found;
}
