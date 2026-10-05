/** mm:ss. Minutes are not wrapped into hours (a 65 minute video is "65:12"). "—" when unknown. */
export function formatDuration(totalSeconds: number | null | undefined): string {
  if (totalSeconds === null || totalSeconds === undefined || !Number.isFinite(totalSeconds) || totalSeconds < 0) return '—'
  const s = Math.floor(totalSeconds)
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let i = 0
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024
    i++
  }
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[i]}`
}

/** Local date and time, e.g. "Mar 4, 2026, 05:06 AM". `locale` / `timeZone` exist so tests are deterministic. */
export function formatDate(iso: string, locale?: string, timeZone?: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleString(locale, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZone })
}

const STAGE_LABELS: Record<string, string> = {
  queued: 'Waiting in queue',
  extracting: 'Extracting audio',
  transcribing: 'Transcribing speech',
  embedding: 'Indexing for search',
  done: 'Ready',
  failed: 'Failed',
}

export function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage
}

export function isTerminalStage(stage: string): boolean {
  return stage === 'done' || stage === 'failed'
}

export type BadgeTone = 'ok' | 'busy' | 'bad' | 'neutral'

/** Video.status: "uploaded" really means "stored, waiting for or in processing". */
export function statusBadge(status: string): { label: string; tone: BadgeTone } {
  switch (status) {
    case 'ready':
      return { label: 'Ready', tone: 'ok' }
    case 'uploaded':
      return { label: 'Processing', tone: 'busy' }
    case 'failed':
      return { label: 'Failed', tone: 'bad' }
    default:
      return { label: status, tone: 'neutral' }
  }
}

/** YouTube-style length: "4:43", "27:48", "1:02:03". "" when unknown. */
export function formatClock(totalSeconds: number | null | undefined): string {
  if (totalSeconds === null || totalSeconds === undefined || !Number.isFinite(totalSeconds) || totalSeconds < 0) return ''
  const s = Math.floor(totalSeconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = String(s % 60).padStart(2, '0')
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`
}

const UNITS: ReadonlyArray<[string, number]> = [['year', 365 * 86400], ['month', 30 * 86400], ['week', 7 * 86400], ['day', 86400], ['hour', 3600], ['minute', 60]]

/** "3 days ago", "1 month ago", "just now". `now` is injectable for tests. "" for an unreadable date. */
export function formatRelative(iso: string, now: number = Date.now()): string {
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const seconds = Math.max(0, Math.floor((now - then) / 1000))
  for (const [name, size] of UNITS) {
    if (seconds >= size) {
      const n = Math.floor(seconds / size)
      return `${n} ${name}${n === 1 ? '' : 's'} ago`
    }
  }
  return 'just now'
}
