import { useCallback, useEffect, useRef, useState } from 'react'
import { describeError } from '../api/errors'
import { streamUrls } from '../api/streams'
import { formatDuration } from '../lib/format'
import styles from './VideoPlayer.module.css'

/** Ask the player to show a moment of a video. A new `nonce` repeats the request even for the same time (seek again). */
export interface SeekRequest {
  videoId: string
  t: number
  play: boolean
  nonce: number
}

interface Props {
  request: SeekRequest | null
  /** Accessible name of the video element. */
  label?: string
  /** Playback position, from the video's timeupdate/seeked events. */
  onTime?: (seconds: number) => void
  emptyMessage?: string
}

type Phase = 'idle' | 'loading' | 'ready' | 'error'

/**
 * One <video> that can be pointed at any video: it fetches the presigned URL (cached per video, see lib/streamUrls.ts),
 * loads it, jumps to the requested second and plays. Asking for another moment of the video that is already loaded
 * just seeks. If the URL turns out to have expired it fetches a fresh one once and resumes where it was.
 */
export default function VideoPlayer({ request, label = 'Video player', onTime, emptyMessage = 'Pick something to play it here.' }: Props) {
  const video = useRef<HTMLVideoElement>(null)
  const loaded = useRef<string | null>(null) // video id whose URL is in `src`
  const pending = useRef<{ t: number; play: boolean } | null>(null)
  const latest = useRef(0) // guards against out-of-order URL responses
  const retried = useRef(false)
  const phaseRef = useRef<Phase>('idle')
  const [src, setSrc] = useState<string | undefined>()
  const [phase, setPhaseState] = useState<Phase>('idle')
  const [message, setMessage] = useState<string | null>(null)
  const [tapToPlay, setTapToPlay] = useState<number | null>(null)

  const setPhase = useCallback((p: Phase) => {
    phaseRef.current = p
    setPhaseState(p)
  }, [])

  const applyPending = useCallback(() => {
    const v = video.current
    const p = pending.current
    if (!v || !p) return
    pending.current = null
    const end = Number.isFinite(v.duration) ? Math.max(0, v.duration - 0.25) : p.t // never park on the very last frame
    v.currentTime = Math.max(0, Math.min(p.t, end))
    if (p.play) {
      v.play().then(
        () => setTapToPlay(null),
        () => setTapToPlay(p.t), // the browser blocked autoplay (no click yet): show a hint, controls still work
      )
    }
  }, [])

  const load = useCallback((videoId: string, fresh: boolean) => {
    const id = ++latest.current
    setPhase('loading')
    setMessage(null)
    if (fresh) streamUrls.invalidate(videoId)
    streamUrls.get(videoId).then(
      (url) => {
        if (id !== latest.current) return
        loaded.current = videoId
        setSrc(url) // the element loads it; loadedmetadata applies the pending seek
      },
      (err: unknown) => {
        if (id !== latest.current) return
        setPhase('error')
        setMessage(describeError(err, 'stream'))
      },
    )
  }, [setPhase])

  useEffect(() => {
    if (!request) return
    pending.current = { t: request.t, play: request.play }
    setTapToPlay(null)
    if (loaded.current === request.videoId && phaseRef.current !== 'error') {
      if ((video.current?.readyState ?? 0) >= 1) applyPending() // already loaded: just seek
      return // otherwise loadedmetadata is still coming and will apply it
    }
    retried.current = false
    load(request.videoId, false)
  }, [request, applyPending, load])

  function onError() {
    const v = video.current
    const id = loaded.current
    if (!v || !id) return
    if (!retried.current) {
      retried.current = true // most likely an expired presigned URL: get a new one and resume
      pending.current = { t: v.currentTime, play: !v.paused }
      load(id, true)
      return
    }
    setPhase('error')
    setMessage("The video couldn't be played.")
  }

  function retry() {
    const id = loaded.current ?? request?.videoId
    if (!id) return
    retried.current = false
    pending.current = pending.current ?? (request ? { t: request.t, play: request.play } : null)
    load(id, true)
  }

  return (
    <div className={styles.frame}>
      {src && (
        <video
          ref={video}
          className={styles.video}
          src={src}
          controls
          playsInline
          preload="metadata"
          aria-label={label}
          onLoadedMetadata={() => {
            setPhase('ready')
            applyPending()
          }}
          onPlay={() => setTapToPlay(null)}
          onTimeUpdate={(e) => onTime?.(e.currentTarget.currentTime)}
          onSeeked={(e) => onTime?.(e.currentTarget.currentTime)}
          onError={onError}
        />
      )}
      {!request && !src && <div className={styles.overlay}>{emptyMessage}</div>}
      {phase === 'loading' && <div className={styles.overlay} role="status">Loading video…</div>}
      {phase === 'error' && (
        <div className={styles.overlay} role="alert">
          <span>{message}</span>
          <button type="button" className="btn btn-primary" onClick={retry}>Try again</button>
        </div>
      )}
      {tapToPlay !== null && phase === 'ready' && (
        <div className={styles.hint}>Press play to start at {formatDuration(tapToPlay)}</div>
      )}
    </div>
  )
}
