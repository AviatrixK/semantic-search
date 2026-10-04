import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { Link } from 'react-router-dom'
import type { Citation } from '../api'
import AgentTrace from '../components/AgentTrace'
import AnswerText, { citationKey } from '../components/AnswerText'
import VideoPlayer from '../components/VideoPlayer'
import type { SeekRequest } from '../components/VideoPlayer'
import { useChat } from '../hooks/useChat'
import { clock } from '../lib/answerText'
import { routeLabel } from '../lib/trace'
import { watchPath } from '../lib/timeParam'
import styles from './Ask.module.css'

const MAX_QUESTION = 1000
const EXAMPLES = [
  'What advice does the speaker give about sounding confident?',
  'Summarize what is said about body language.',
  'What mistakes should I avoid when speaking?',
]

export default function Ask() {
  const chat = useChat()
  const [input, setInput] = useState('')
  const [selected, setSelected] = useState<Citation | null>(null) // the source loaded in the player
  const [request, setRequest] = useState<SeekRequest | null>(null)
  const nonce = useRef(0)
  const playerBox = useRef<HTMLElement>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const box = useRef<HTMLTextAreaElement>(null)

  const trimmed = input.trim()
  const blocked = chat.secondsBlocked > 0
  const canSend = trimmed.length >= 3 && !chat.pending && !blocked

  // keep the newest message in view
  const last = chat.messages.at(-1)
  useEffect(() => {
    const calm = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    endRef.current?.scrollIntoView({ block: 'end', behavior: calm ? 'auto' : 'smooth' })
  }, [chat.messages.length, last?.role === 'assistant' ? last.status : ''])

  const play = useCallback((c: Citation) => {
    setSelected(c)
    setRequest({ videoId: c.video_id, t: c.start_sec, play: true, nonce: ++nonce.current })
    playerBox.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }) // matters when stacked (mobile)
  }, [])

  function submit(e?: FormEvent) {
    e?.preventDefault()
    if (canSend && chat.send(trimmed)) setInput('')
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault() // Enter sends, Shift+Enter adds a new line
      submit()
    }
  }

  function cancel(id: string) {
    setInput(chat.cancel(id)) // the question goes back into the box
    box.current?.focus()
  }

  return (
    <main className={styles.page}>
      <div className={styles.layout}>
        <section className={styles.chat} aria-label="Ask">
          <header className={styles.head}>
            <h1>Ask</h1>
            {chat.messages.length > 0 && (
              <button type="button" className="btn btn-secondary" onClick={chat.clear}>Clear conversation</button>
            )}
          </header>

          <div className={styles.log} role="log" aria-live="polite" aria-label="Conversation">
            {chat.messages.length === 0 && (
              <div className={styles.empty}>
                <p>
                  Ask a question about your videos. Answers are written only from the transcripts, and every claim links to the
                  moment it came from: click a numbered chip to play it.
                </p>
                <div className={styles.examples}>
                  {EXAMPLES.map((q) => (
                    <button key={q} type="button" className={styles.example} onClick={() => { setInput(q); box.current?.focus() }}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {chat.messages.map((m) =>
              m.role === 'user' ? (
                <div key={m.id} className={`${styles.bubble} ${styles.user}`}>
                  <span className="sr-only">You: </span>
                  {m.text}
                </div>
              ) : (
                <div key={m.id} className={`${styles.bubble} ${styles.assistant}`} aria-busy={m.status === 'pending'}>
                  <span className="sr-only">Answer: </span>
                  {m.route && (
                    <span className={styles.route} title={`${routeLabel(m.route)?.hint}${m.routeReason ? ` (${m.routeReason})` : ''}`}>
                      {routeLabel(m.route)?.label}
                    </span>
                  )}
                  {m.status === 'pending' && <AgentTrace steps={m.trace} running />}
                  {m.status === 'pending' && (
                    <div className={styles.thinking}>
                      <span className={styles.dots} aria-hidden="true"><i /><i /><i /></span>
                      <span>{m.route === 'agent' ? 'Researching the videos…' : 'Searching the videos and writing an answer…'}</span>
                      <button type="button" className="btn btn-secondary" onClick={() => cancel(m.id)}>Cancel</button>
                    </div>
                  )}
                  {m.status === 'error' && (
                    <div className={styles.error} role="alert">
                      <span>{m.error}</span>
                      <button type="button" className="btn btn-secondary" onClick={() => chat.retry(m.id)}
                        disabled={chat.pending || blocked}>
                        {blocked ? `Try again in ${chat.secondsBlocked}s` : 'Try again'}
                      </button>
                    </div>
                  )}
                  {m.status === 'error' && m.trace.length > 0 && <AgentTrace steps={m.trace} running={false} />}
                  {m.status === 'done' && (
                    <>
                      <AnswerText text={m.text} citations={m.citations} selectedKey={selected ? citationKey(selected) : null} onCite={play} />
                      {m.citations.length > 0 && (
                        <ul className={styles.sources} aria-label="Sources">
                          {m.citations.map((c) => (
                            <li key={c.n}>
                              <button
                                type="button"
                                className={`${styles.source} ${selected && citationKey(selected) === citationKey(c) ? styles.sourceOn : ''}`}
                                onClick={() => play(c)}
                              >
                                <span className={styles.n}>{c.n}</span>
                                <span className={styles.sourceTitle}>{c.title}</span>
                                <span className={styles.time}>{clock(c.start_sec)}</span>
                              </button>
                            </li>
                          ))}
                        </ul>
                      )}
                      <AgentTrace steps={m.trace} running={false} />
                      {m.usage && m.usage.llmCalls > 0 && (
                        <p className={styles.meta}>
                          {m.usage.llmCalls} AI call{m.usage.llmCalls === 1 ? '' : 's'} · {m.usage.tokens.toLocaleString()} tokens
                        </p>
                      )}
                    </>
                  )}
                </div>
              ),
            )}
            <div ref={endRef} />
          </div>

          <form className={styles.composer} onSubmit={submit}>
            <textarea
              ref={box}
              className={`input ${styles.input}`}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask about something said in the videos…"
              aria-label="Your question"
              rows={2}
              maxLength={MAX_QUESTION}
              autoFocus
            />
            <button type="submit" className="btn btn-primary" disabled={!canSend}>
              {chat.pending ? 'Thinking…' : blocked ? `Wait ${chat.secondsBlocked}s` : 'Ask'}
            </button>
          </form>
          <p className={styles.note}>
            Generated from the video transcripts only, and it can still be wrong: check the cited moments.
            {input.length > MAX_QUESTION - 100 && ` ${MAX_QUESTION - input.length} characters left.`}
          </p>
        </section>

        <aside className={styles.player} ref={playerBox} data-active={request ? '' : undefined} aria-label="Video player">
          <VideoPlayer request={request} label={selected?.title ?? 'Video player'} emptyMessage="Click a numbered source to play it here." />
          {selected && (
            <div className={styles.now}>
              <strong title={selected.title}>{selected.title}</strong>
              <span>Source {selected.n} · {clock(selected.start_sec)} – {clock(selected.end_sec)}</span>
              <Link to={watchPath(selected.video_id, selected.start_sec)}>Open with full transcript</Link>
            </div>
          )}
        </aside>
      </div>
    </main>
  )
}
