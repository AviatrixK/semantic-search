import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { SearchHit, Video } from '../api'
import { formatClock, formatDate, formatRelative } from '../lib/format'
import { foundByBadge, scoreText } from '../lib/searchParams'
import { chooseSnippet, isNonEnglishVideo, languageLabel, markSegments, queryTerms } from '../lib/snippet'
import type { SnippetPart } from '../lib/snippet'
import { linkLangForQuery } from '../lib/transcriptLang'
import type { SearchMode } from '../lib/searchParams'
import { watchPath } from '../lib/timeParam'
import VideoThumb from './VideoThumb'
import styles from './ResultCard.module.css'

interface Props {
  hit: SearchHit
  /** The hit's video, when known (gives the length and the upload date). */
  video?: Video
  /** The mode this result came from: names the score bar (keyword scores are not similarities). */
  mode: SearchMode
  /** The query that produced the hit: decides which version of the text comes first and which words are marked. */
  query: string
}

function Marked({ part, terms, highlight }: { part: SnippetPart; terms: string[]; highlight: string | null }) {
  return (
    <>
      {markSegments(part.text, terms, { kind: part.kind, highlight }).map((s, i) => (s.hit ? <mark key={i}>{s.text}</mark> : s.text))}
    </>
  )
}

/**
 * One search result, YouTube style: the thumbnail is a colour tile (no video is loaded until it is opened), then the title, upload info, a
 * "jump to" time chip and the matching sentence in bold. The whole row opens the video at that moment.
 */
export default function ResultCard({ hit, video, mode, query }: Props) {
  const [swapped, setSwapped] = useState(false)
  const { primary, secondary } = chooseSnippet(hit, query, swapped)
  const terms = queryTerms(query)
  const language = languageLabel(hit.language)
  const pct = Math.round(Math.max(0, Math.min(1, hit.score)) * 100)
  const badge = mode === 'hybrid' ? foundByBadge(hit.found_by) : null
  const score = scoreText(mode, hit.score)
  const uploaded = video ? formatRelative(video.created_at) || formatDate(video.created_at) : ''
  return (
    <Link to={watchPath(hit.video_id, hit.start_sec, linkLangForQuery(query, isNonEnglishVideo(hit.language, hit.text)))} className={styles.row}>
      <div className={styles.thumb}>
        <VideoThumb videoId={hit.video_id} duration={video?.duration_sec} />
      </div>
      <div className={styles.body}>
        <h2 className={styles.title}>{hit.title}</h2>
        <p className={styles.meta}>{[uploaded && `Uploaded ${uploaded}`, video && formatClock(video.duration_sec)].filter(Boolean).join(' · ')}</p>
        <p className={styles.moment}>
          <span className={styles.chip}>Jump to {formatClock(hit.start_sec)}</span>
          <span className={styles.range}>{formatClock(hit.start_sec)} – {formatClock(hit.end_sec)}</span>
        </p>
        <p className={styles.text} lang={primary.kind === 'english' ? 'en' : undefined}>
          <Marked part={primary} terms={terms} highlight={hit.highlight} />
        </p>
        {secondary && (
          <p className={styles.original} lang={secondary.kind === 'english' ? 'en' : undefined}>
            <Marked part={secondary} terms={terms} highlight={hit.highlight} />
          </p>
        )}
        <div className={styles.foot}>
          {language && <span className={styles.lang} title={`Spoken in ${language}`}>{language}</span>}
          {secondary && (
            <button
              type="button"
              className={styles.toggle}
              onClick={(e) => {
                e.preventDefault() // the whole card is a link: this only switches the text
                e.stopPropagation()
                setSwapped((s) => !s)
              }}
            >
              {primary.kind === 'english' ? 'Show original' : 'Show English'}
            </button>
          )}
          {badge && <span className={styles.found} title={badge.title}>{badge.label}</span>}
          <span
            className={styles.score}
            role="meter"
            aria-label={score.label}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct}
            aria-valuetext={`${pct}% match`}
            title={score.title}
          >
            <span style={{ width: `${pct}%` }} />
          </span>
        </div>
      </div>
    </Link>
  )
}
