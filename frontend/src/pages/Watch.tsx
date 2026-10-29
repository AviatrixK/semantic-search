import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import Spinner from '../components/Spinner'
import TranscriptLangToggle from '../components/TranscriptLangToggle'
import TranscriptPanel from '../components/TranscriptPanel'
import VideoPlayer from '../components/VideoPlayer'
import type { SeekRequest } from '../components/VideoPlayer'
import { useTranscript } from '../hooks/useTranscript'
import { useVideos } from '../hooks/useVideos'
import { formatClock, formatDate, formatRelative } from '../lib/format'
import { parseTimeParam, watchPath } from '../lib/timeParam'
import { isNonEnglishVideo } from '../lib/snippet'
import { activeLineIndex, buildTranscript, seekRequest } from '../lib/transcript'
import type { TranscriptLang, TranscriptLine } from '../lib/transcript'
import { initialTranscriptLang, readSavedTranscriptLang, saveTranscriptLang } from '../lib/transcriptLang'
import styles from './Watch.module.css'

/** /watch/:videoId?t=123: the video with its full transcript. `t` (seconds, 2:03 or 1m30s) starts it at that moment. */
export default function Watch() {
  const { videoId = '' } = useParams()
  const [params] = useSearchParams()
  const tParam = parseTimeParam(params.get('t'))
  const { videos, loading, error: listError, retry } = useVideos()
  const video = videos?.find((v) => v.id === videoId)
  const transcript = useTranscript(videoId)
  // Non-English videos get an Original | English | Romanized toggle. The user's saved choice wins, then ?lang= from the search link.
  const toggleShown = transcript.status === 'ready' && transcript.chunks.length > 0 && isNonEnglishVideo(transcript.language, transcript.chunks[0].text)
  const [saved] = useState(() => readSavedTranscriptLang())
  const [picked, setPicked] = useState<TranscriptLang | null>(null)
  const lang = toggleShown ? picked ?? initialTranscriptLang({ toggleShown, saved, urlLang: params.get('lang') }) : 'original'
  const transcriptLines = useMemo(() => buildTranscript(transcript.chunks, lang), [transcript.chunks, lang])
  const pickLang = useCallback((next: TranscriptLang) => {
    setPicked(next)
    saveTranscriptLang(next) // best effort: a blocked localStorage only means the choice is not remembered
  }, [])

  const [request, setRequest] = useState<SeekRequest | null>(null)
  const nonce = useRef(0)
  const [active, setActive] = useState(-1)
  const time = useRef(0)
  const lines = useRef<TranscriptLine[]>([])
  const [copied, setCopied] = useState<string | null>(null)

  // Open (or re-open, when the link changes) at the requested moment. Without ?t= the video just loads, paused at 0.
  useEffect(() => {
    time.current = tParam ?? 0
    setRequest({ videoId, t: tParam ?? 0, play: tParam !== null, nonce: ++nonce.current })
  }, [videoId, tParam])

  useEffect(() => {
    lines.current = transcriptLines
    setActive(activeLineIndex(transcriptLines, time.current))  // the same moment stays highlighted whichever language is shown
  }, [transcriptLines])

  const onTime = useCallback((t: number) => {
    time.current = t
    const i = activeLineIndex(lines.current, t)
    setActive((cur) => (cur === i ? cur : i)) // only re-render when the spoken line changes
  }, [])

  const seekTo = useCallback((line: TranscriptLine) => {
    setRequest(seekRequest(videoId, line, ++nonce.current))
  }, [videoId])

  async function copyLink() {
    const url = `${window.location.origin}${watchPath(videoId, time.current)}`
    try {
      await navigator.clipboard.writeText(url)
      setCopied('Link copied')
    } catch {
      setCopied(url) // clipboard blocked: show the link so it can be copied by hand
    }
    setTimeout(() => setCopied(null), 4000)
  }

  const notFound = videos !== null && !video && !loading
  if (notFound) {
    return (
      <main className={styles.page}>
        <h1>Video not found</h1>
        <p className={styles.muted}>It may have been deleted.</p>
        <Link to="/library">Back to the library</Link>
      </main>
    )
  }

  return (
    <main className={styles.page}>
      <div className={styles.layout}>
        <section className={styles.playerCol} aria-label="Video">
          <VideoPlayer request={request} label={video?.title ?? 'Video player'} onTime={onTime} emptyMessage="Loading…" />
          <div className={styles.info}>
            <h1 title={video?.title}>{video?.title ?? 'Loading…'}</h1>
            <p className={styles.muted}>
              {video && [formatClock(video.duration_sec), `Uploaded ${formatRelative(video.created_at) || formatDate(video.created_at)}`].filter(Boolean).join(' · ')}
            </p>
            <div className={styles.actions}>
              <button type="button" className="btn btn-secondary" onClick={() => void copyLink()}>
                Copy link to this moment
              </button>
              {video?.status === 'ready' && (
                <Link className="btn btn-secondary" to={`/search?${new URLSearchParams({ video: videoId })}`}>Search in this video</Link>
              )}
              {copied && <span className={styles.copied} role="status">{copied}</span>}
            </div>
            {video && video.status !== 'ready' && (
              <p className={styles.warn} role="status">
                {video.status === 'failed'
                  ? 'Processing this video failed, so it has no transcript.'
                  : 'This video is still being processed. The transcript will appear when it is ready.'}
              </p>
            )}
            {listError && !video && (
              <p className={styles.warn} role="alert">
                {listError} <button type="button" className="btn btn-secondary" onClick={() => void retry()}>Try again</button>
              </p>
            )}
          </div>
        </section>

        <section className={styles.transcriptCol} aria-label="Transcript">
          <h2>Transcript</h2>
          {transcript.status === 'loading' && <Spinner label="Loading transcript…" />}
          {transcript.status === 'error' && (
            <div className={styles.error} role="alert">
              <span>{transcript.error}</span>
              <button type="button" className="btn btn-secondary" onClick={transcript.retry}>Try again</button>
            </div>
          )}
          {transcript.status === 'ready' && transcriptLines.length === 0 && (
            <p className={styles.muted}>No transcript yet. The video may still be processing, or it has no speech.</p>
          )}
          {toggleShown && <TranscriptLangToggle value={lang} onChange={pickLang} />}
          {transcript.status === 'ready' && transcriptLines.length > 0 && (
            <TranscriptPanel lines={transcriptLines} activeIndex={active} onSelect={seekTo} />
          )}
        </section>
      </div>
    </main>
  )
}
