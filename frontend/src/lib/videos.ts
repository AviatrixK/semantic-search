export type VideoSort = 'newest' | 'oldest' | 'longest' | 'shortest'

export const SORT_OPTIONS: ReadonlyArray<{ value: VideoSort; label: string }> = [
  { value: 'newest', label: 'Newest' },
  { value: 'oldest', label: 'Oldest' },
  { value: 'longest', label: 'Longest' },
  { value: 'shortest', label: 'Shortest' },
]

interface Sortable {
  created_at: string
  duration_sec: number | null
}

const time = (iso: string) => {
  const t = new Date(iso).getTime()
  return Number.isNaN(t) ? 0 : t
}

/** A sorted copy (the input is left alone). Ties keep their incoming order; a video with no known duration goes last. */
export function sortVideos<T extends Sortable>(videos: readonly T[], sort: VideoSort): T[] {
  const indexed = videos.map((v, i) => ({ v, i }))
  const byDuration = (dir: 1 | -1) => (a: { v: T; i: number }, b: { v: T; i: number }) => {
    const da = a.v.duration_sec
    const db = b.v.duration_sec
    if (da === null && db === null) return a.i - b.i
    if (da === null) return 1
    if (db === null) return -1
    return (da - db) * dir || a.i - b.i
  }
  const compare =
    sort === 'oldest' ? (a: { v: T; i: number }, b: { v: T; i: number }) => time(a.v.created_at) - time(b.v.created_at) || a.i - b.i
    : sort === 'longest' ? byDuration(-1)
    : sort === 'shortest' ? byDuration(1)
    : (a: { v: T; i: number }, b: { v: T; i: number }) => time(b.v.created_at) - time(a.v.created_at) || a.i - b.i
  return indexed.sort(compare).map((x) => x.v)
}

/** A stable 0-359 colour angle for a video id: its placeholder thumbnail always has the same colours. */
export function hueFor(id: string): number {
  let h = 0
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) % 360
  return h
}
