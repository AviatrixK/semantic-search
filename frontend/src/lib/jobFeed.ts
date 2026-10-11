import { ApiError } from '../api/errors.js'
import type { Job } from '../api/types.js'
import { isTerminalStage } from './format.js'
import type { SseEvent } from './sse.js'

/** What GET /api/jobs/events sends: one `snapshot` (every active job) on connect, then a `job` event per change. */
export type JobFeedEvent = { type: 'snapshot'; jobs: Job[] } | { type: 'job'; job: Job }

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

function toJob(v: unknown): Job | null {
  if (!isObj(v)) return null
  const { id, video_id, stage, progress, error } = v
  if (typeof id !== 'string' || typeof video_id !== 'string' || typeof stage !== 'string') return null
  return {
    id,
    video_id,
    stage,
    progress: typeof progress === 'number' && Number.isFinite(progress) ? Math.max(0, Math.min(100, progress)) : 0,
    error: typeof error === 'string' ? error : null,
  }
}

/** One SSE event -> a typed one, or null for anything unknown or malformed (a bad event must never crash the page). */
export function parseJobEvent(e: SseEvent): JobFeedEvent | null {
  let data: unknown
  try {
    data = JSON.parse(e.data)
  } catch {
    return null
  }
  if (e.event === 'snapshot') {
    if (!isObj(data) || !Array.isArray(data.jobs)) return null
    return { type: 'snapshot', jobs: data.jobs.map(toJob).filter((j): j is Job => j !== null) }
  }
  if (e.event === 'job') {
    const job = toJob(data)
    return job ? { type: 'job', job } : null
  }
  return null
}

// ------------------------------------------------------------------ state

export type JobMap = Readonly<Record<string, Job>>

export type JobAction =
  | { type: 'snapshot'; jobs: Job[] }
  | { type: 'job'; job: Job }
  | { type: 'dismiss'; id: string }

export const isActive = (job: Job): boolean => !isTerminalStage(job.stage)

export function hasActiveJob(jobs: JobMap): boolean {
  return Object.values(jobs).some(isActive)
}

/**
 * snapshot: the server's answer is the truth about running jobs, so a job we believed was running but is missing from it has
 * finished while we were disconnected: it is dropped (the caller reloads the video list). Finished jobs we already know stay
 * until dismissed. job: a newer state replaces the older one; a late `queued` can never undo `done`.
 */
export function jobsReducer(state: JobMap, action: JobAction): JobMap {
  switch (action.type) {
    case 'snapshot': {
      const next: Record<string, Job> = {}
      for (const [id, job] of Object.entries(state)) if (!isActive(job)) next[id] = job
      for (const job of action.jobs) if (!(job.id in next)) next[job.id] = job
      return next
    }
    case 'job': {
      const known = state[action.job.id]
      if (known && !isActive(known) && isActive(action.job)) return state
      return { ...state, [action.job.id]: action.job }
    }
    case 'dismiss': {
      if (!(action.id in state)) return state
      const { [action.id]: _gone, ...rest } = state
      return rest
    }
  }
}

/** Active jobs in `before` that `after` no longer lists as active: they ended while the stream was down. */
export function endedWhileAway(before: JobMap, snapshot: Job[]): boolean {
  const running = new Set(snapshot.map((j) => j.id))
  return Object.values(before).some((j) => isActive(j) && !running.has(j.id))
}

// ------------------------------------------------------------------ connection

export type FeedStatus = 'idle' | 'connecting' | 'live' | 'reconnecting' | 'stopped'

export interface FeedOptions {
  /** api.stream: resolves when the server ends the stream, rejects on network errors, ApiError before it starts. */
  stream: (path: string, o: { onEvent: (e: SseEvent) => void; signal: AbortSignal }) => Promise<void>
  onEvent: (e: JobFeedEvent) => void
  onStatus: (s: FeedStatus) => void
  path?: string
  baseDelayMs?: number
  maxDelayMs?: number
  /** Resolves after `ms`, or at once when aborted. Injectable for tests. */
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>
}

export const FEED_PATH = '/api/jobs/events'
const FATAL = new Set([401, 403, 404])

const defaultSleep = (ms: number, signal: AbortSignal): Promise<void> =>
  new Promise((resolve) => {
    if (signal.aborted) return resolve()
    const id = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => { clearTimeout(id); resolve() }, { once: true })
  })

/** Waiting time before reconnect attempt number `attempt` (1, 2, 3...): 1 s, 2 s, 4 s ... capped. */
export function reconnectDelay(attempt: number, base = 1000, max = 30000): number {
  return Math.min(max, base * 2 ** Math.max(0, attempt - 1))
}

/**
 * Holds ONE Server-Sent Events connection open and re-opens it, with growing pauses, only if it drops. This is a
 * reconnect, not polling: while the connection is healthy there are no requests at all. Stops for good on 401/403/404 (the
 * session ended or the user is not an admin). The returned function closes the connection.
 */
export function connectJobFeed(o: FeedOptions): () => void {
  const controller = new AbortController()
  const { signal } = controller
  const sleep = o.sleep ?? defaultSleep

  void (async () => {
    let attempt = 0
    while (!signal.aborted) {
      o.onStatus(attempt === 0 ? 'connecting' : 'reconnecting')
      try {
        await o.stream(o.path ?? FEED_PATH, {
          signal,
          onEvent: (raw) => {
            const event = parseJobEvent(raw)
            if (!event) return
            if (event.type === 'snapshot') {
              attempt = 0 // the connection works: the next drop starts from a short pause again
              o.onStatus('live')
            }
            o.onEvent(event)
          },
        })
      } catch (error) {
        if (signal.aborted) return
        if (error instanceof ApiError && FATAL.has(error.status)) {
          o.onStatus('stopped')
          return
        }
      }
      if (signal.aborted) return
      attempt++
      o.onStatus('reconnecting')
      await sleep(reconnectDelay(attempt, o.baseDelayMs, o.maxDelayMs), signal)
    }
  })()

  return () => controller.abort()
}
