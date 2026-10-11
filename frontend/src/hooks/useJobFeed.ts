import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { api } from '../api'
import type { Job } from '../api'
import { connectJobFeed, endedWhileAway, jobsReducer } from '../lib/jobFeed'
import type { FeedStatus, JobMap } from '../lib/jobFeed'

interface Handlers {
  /** A job reached done/failed: the video list has changed (once, no polling). */
  onFinished?: (job: Job) => void
  /** The stream was down and jobs ended meanwhile (or it was just re-established): reload the video list. */
  onResync?: () => void
}

export interface JobFeed {
  jobs: JobMap
  status: FeedStatus
  dismiss: (jobId: string) => void
}

/**
 * Live job progress from GET /api/jobs/events (Server-Sent Events). The connection is open only while `enabled`: the admin page
 * passes true while at least one job is active and false otherwise, so an idle page holds no connection and makes no requests.
 * Every connect starts with a snapshot of the running jobs, so a page refresh shows the right progress at once.
 */
export function useJobFeed(enabled: boolean, handlers: Handlers = {}): JobFeed {
  const [jobs, dispatch] = useReducer(jobsReducer, {} as JobMap)
  const [status, setStatus] = useState<FeedStatus>('idle')
  const jobsRef = useRef<JobMap>(jobs)
  jobsRef.current = jobs
  const handlersRef = useRef(handlers)
  handlersRef.current = handlers

  useEffect(() => {
    if (!enabled) {
      setStatus('idle')
      return
    }
    let first = true
    const stop = connectJobFeed({
      stream: (path, o) => api.stream(path, o),
      onStatus: setStatus,
      onEvent: (e) => {
        if (e.type === 'snapshot') {
          if (!first || endedWhileAway(jobsRef.current, e.jobs)) handlersRef.current.onResync?.()
          first = false
          dispatch({ type: 'snapshot', jobs: e.jobs })
        } else {
          dispatch({ type: 'job', job: e.job })
          if (e.job.stage === 'done' || e.job.stage === 'failed') handlersRef.current.onFinished?.(e.job)
        }
      },
    })
    return () => {
      stop() // leaving the page, or the last job finished: close the connection
      setStatus('idle')
    }
  }, [enabled])

  return { jobs, status, dismiss: useCallback((id: string) => dispatch({ type: 'dismiss', id }), []) }
}
