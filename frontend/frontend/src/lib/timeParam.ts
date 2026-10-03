const MAX_SECONDS = 100 * 3600

/**
 * Parses the `t` of a deep link such as /watch/<id>?t=123. Accepts seconds ("123", "12.5"), clock times ("2:03",
 * "1:02:03") and YouTube-style ("90s", "1m30s", "1h2m3s"). Returns null for anything else (negative, huge, junk).
 */
export function parseTimeParam(value: string | null | undefined): number | null {
  if (value === null || value === undefined) return null
  const s = value.trim().toLowerCase()
  if (s === '') return null

  let seconds: number | null = null
  let m: RegExpExecArray | null
  if (/^\d+(\.\d+)?$/.test(s)) {
    seconds = Number(s)
  } else if ((m = /^(\d+):([0-5]?\d(?:\.\d+)?)$/.exec(s))) {
    seconds = Number(m[1]) * 60 + Number(m[2])
  } else if ((m = /^(\d+):([0-5]?\d):([0-5]?\d(?:\.\d+)?)$/.exec(s))) {
    seconds = Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3])
  } else if ((m = /^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?$/.exec(s)) && (m[1] || m[2] || m[3])) {
    seconds = Number(m[1] ?? 0) * 3600 + Number(m[2] ?? 0) * 60 + Number(m[3] ?? 0)
  }
  return seconds !== null && Number.isFinite(seconds) && seconds <= MAX_SECONDS ? seconds : null
}

/** Relative link to a moment of a video, e.g. /watch/abc?t=123 (whole seconds). */
export function watchPath(videoId: string, seconds?: number): string {
  const base = `/watch/${encodeURIComponent(videoId)}`
  return seconds === undefined ? base : `${base}?t=${Math.max(0, Math.floor(seconds))}`
}
