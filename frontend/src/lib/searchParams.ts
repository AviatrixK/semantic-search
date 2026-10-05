/** Search options (mode and filters): what the URL and the request carry, validated so a hand-edited link cannot break the page. */
export const SEARCH_MODES = ['hybrid', 'vector', 'keyword'] as const
export type SearchMode = (typeof SEARCH_MODES)[number]

export interface SearchFilters {
  mode: SearchMode
  /** Only this video (a UUID). */
  videoId: string | null
  /** Only videos uploaded on or after this day, YYYY-MM-DD (UTC). */
  after: string | null
}

export const DEFAULT_FILTERS: SearchFilters = { mode: 'hybrid', videoId: null, after: null }

export const MODE_OPTIONS: ReadonlyArray<{ value: SearchMode; label: string; hint: string }> = [
  { value: 'hybrid', label: 'Best match', hint: 'Meaning and exact words together. Recommended.' },
  { value: 'vector', label: 'Meaning', hint: 'Finds ideas even when the speaker used different words.' },
  { value: 'keyword', label: 'Exact words', hint: 'Finds the words you type, such as names and acronyms. Needs every word to appear.' },
]

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

/** A real calendar day in YYYY-MM-DD form ("2024-02-30" is not). */
export function isValidDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const d = new Date(`${value}T00:00:00Z`)
  return !Number.isNaN(d.getTime()) && d.toISOString().slice(0, 10) === value
}

export function parseFilters(params: { get(name: string): string | null }): SearchFilters {
  const mode = params.get('mode')
  const video = params.get('video')
  const after = params.get('after')
  return {
    mode: SEARCH_MODES.includes(mode as SearchMode) ? (mode as SearchMode) : 'hybrid',
    videoId: video && UUID.test(video) ? video.toLowerCase() : null,
    after: after && isValidDate(after) ? after : null,
  }
}

/** The URL parameters for a set of filters. Defaults are left out to keep shared links short. */
export function filtersToParams(f: SearchFilters): Record<string, string> {
  const out: Record<string, string> = {}
  if (f.mode !== 'hybrid') out.mode = f.mode
  if (f.videoId) out.video = f.videoId
  if (f.after) out.after = f.after
  return out
}

/** Two filter sets with the same key give the same results. */
export function filtersKey(f: SearchFilters): string {
  return `${f.mode}|${f.videoId ?? ''}|${f.after ?? ''}`
}

/** The mode a filtersKey() was made from (what the results on screen were searched with). */
export function modeOfKey(key: string): SearchMode {
  const mode = key.split('|')[0]
  return SEARCH_MODES.includes(mode as SearchMode) ? (mode as SearchMode) : 'hybrid'
}

/** How many of the optional filters (video, date) are on; the mode is shown separately. */
export function activeFilterCount(f: SearchFilters): number {
  return (f.videoId ? 1 : 0) + (f.after ? 1 : 0)
}

/** The address of the results page for a query (what the header search box and result links use). */
export function searchPagePath(query: string, f: SearchFilters = DEFAULT_FILTERS): string {
  const p = new URLSearchParams()
  if (query.trim()) p.set('q', query.trim())
  for (const [k, v] of Object.entries(filtersToParams(f))) p.set(k, v)
  const qs = p.toString()
  return qs ? `/search?${qs}` : '/search'
}

export function searchUrl(query: string, k: number, f: SearchFilters): string {
  const p = new URLSearchParams({ q: query, k: String(k), mode: f.mode })
  if (f.videoId) p.set('video_id', f.videoId)
  if (f.after) p.set('uploaded_after', f.after)
  return `/api/search?${p}`
}

/** Which search found a hybrid result: shown as a small badge so the modes can be compared by eye. */
export function foundByBadge(foundBy: readonly string[] | null | undefined): { label: string; title: string } | null {
  if (!foundBy || foundBy.length === 0) return null
  const meaning = foundBy.includes('vector')
  const words = foundBy.includes('keyword')
  if (meaning && words) return { label: 'Meaning + words', title: 'Found by both the meaning search and the exact-word search' }
  if (meaning) return { label: 'Meaning', title: 'Found by the meaning search only' }
  if (words) return { label: 'Exact words', title: 'Found by the exact-word search only' }
  return null
}

/** The relevance bar's name: keyword results have a keyword score, the others a cosine similarity. */
export function scoreText(mode: SearchMode, score: number): { label: string; title: string } {
  const label = mode === 'keyword' ? 'Keyword relevance' : 'Similarity'
  return { label, title: `${label} ${score.toFixed(2)}` }
}
