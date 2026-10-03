import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Job } from '../api'
import { ApiError, describeError } from '../api/errors'
import { isTerminalStage } from '../lib/format'
import { startPolling } from '../lib/poller'

export interface JobPolling {
  job: Job | null
  /** Set while polling hits trouble (e.g. the backend is briefly unreachable); cleared by the next good response. */
  problem: string | null
  /** True when polling stopped because retrying is pointless (job gone, session ended). */
  gaveUp: boolean
  /** The job reached done/failed: polling has stopped. */
  finished: boolean
}

const FATAL = new Set([401, 403, 404])

/** Polls GET /api/jobs/{id} every `intervalMs` (default 2 s), one request at a time, until done/failed. */
export function useJobPolling(jobId: string, intervalMs = 2000): JobPolling {
  const [job, setJob] = useState<Job | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [gaveUp, setGaveUp] = useState(false)

  useEffect(() => {
    setJob(null)
    setProblem(null)
    setGaveUp(false)
    return startPolling<Job>({
      intervalMs,
      fetch: (signal) => api.request<Job>(`/api/jobs/${jobId}`, { signal }),
      isDone: (j) => isTerminalStage(j.stage),
      onUpdate: (j) => {
        setJob(j)
        setProblem(null)
      },
      onError: (err, info) => {
        setProblem(describeError(err, 'job'))
        setGaveUp(info.fatal)
      },
      isFatal: (err) => err instanceof ApiError && FATAL.has(err.status),
    }) // the returned stop function runs on unmount or when jobId changes
  }, [jobId, intervalMs])

  return { job, problem, gaveUp, finished: job !== null && isTerminalStage(job.stage) }
}
