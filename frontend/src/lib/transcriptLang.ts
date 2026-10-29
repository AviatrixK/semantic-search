import type { TranscriptLang } from './transcript.js'
import { scriptOf } from './snippet.js'

export const TRANSCRIPT_LANG_KEY = 'svs.transcriptLang'

export const TRANSCRIPT_LANGS: readonly { value: TranscriptLang; label: string }[] = [
  { value: 'original', label: 'Original' },
  { value: 'en', label: 'English' },
  { value: 'roman', label: 'Romanized' },
]

/** The language named by a `?lang=` value or a stored string, or null for anything else. */
export function parseTranscriptLang(value: unknown): TranscriptLang | null {
  return value === 'original' || value === 'en' || value === 'roman' ? value : null
}

type ReadableStorage = Pick<Storage, 'getItem'>
type WritableStorage = Pick<Storage, 'setItem'>

/** localStorage, or undefined where it does not exist or throws on access (private mode, blocked site data). */
function browserStorage(): Storage | undefined {
  try {
    return globalThis.localStorage
  } catch {
    return undefined
  }
}

/** The language the user chose last time, or null (nothing saved, or the storage is unavailable). Never throws. */
export function readSavedTranscriptLang(storage: ReadableStorage | undefined = browserStorage()): TranscriptLang | null {
  try {
    return parseTranscriptLang(storage?.getItem(TRANSCRIPT_LANG_KEY))
  } catch {
    return null
  }
}

/** Remembers the user's choice. Returns false (and the page carries on) when the storage refuses. */
export function saveTranscriptLang(lang: TranscriptLang, storage: WritableStorage | undefined = browserStorage()): boolean {
  try {
    if (!storage) return false
    storage.setItem(TRANSCRIPT_LANG_KEY, lang)
    return true
  } catch {
    return false
  }
}

/**
 * Which language the transcript opens in.
 *  - An English-speaking video (no toggle): always the original.
 *  - Otherwise the user's own earlier choice wins, then `?lang=` from the link they followed (a search in Latin letters adds ?lang=en),
 *    and by default the original.
 */
export function initialTranscriptLang(opts: { toggleShown: boolean; saved: TranscriptLang | null; urlLang: string | null }): TranscriptLang {
  if (!opts.toggleShown) return 'original'
  return opts.saved ?? parseTranscriptLang(opts.urlLang) ?? 'original'
}

/** `?lang=` for a search result link: "en" when the query is in Latin letters (English or Hinglish) and the video is not English. */
export function linkLangForQuery(query: string, videoIsNonEnglish: boolean): 'en' | undefined {
  return videoIsNonEnglish && scriptOf(query) === 'latin' ? 'en' : undefined
}
