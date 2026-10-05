import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { api } from '../api'
import { ApiError, describeError } from '../api/errors'
import { parseAskEvent } from '../lib/askStream'
import {
  CHAT_STORAGE_KEY, SESSION_STORAGE_KEY, chatReducer, isPending, isValidSessionId, newSessionId, restoreMessages, serializeMessages,
} from '../lib/chat'
import type { ChatState } from '../lib/chat'
import { fromServerTrace } from '../lib/trace'
import { useCountdown } from './useCountdown'

/** The backend allows up to 3 model calls of 30 s each per step (retries on 429/5xx) and several steps; give up after this. */
const ASK_TIMEOUT_MS = 120_000
let counter = 0
const newId = () => `m${Date.now().toString(36)}-${++counter}`

function secureRandom(): number {
  const c = globalThis.crypto
  if (c?.getRandomValues) return c.getRandomValues(new Uint32Array(1))[0] / 2 ** 32
  return Math.random()
}

function loadInitial(): ChatState {
  try {
    return { messages: restoreMessages(sessionStorage.getItem(CHAT_STORAGE_KEY)) }
  } catch {
    return { messages: [] } // storage blocked (private mode, disabled cookies)
  }
}

/** The id the server remembers this chat's last turns under. Kept with the conversation, replaced when it is cleared. */
function loadSessionId(): string {
  try {
    const stored = sessionStorage.getItem(SESSION_STORAGE_KEY)
    if (isValidSessionId(stored)) return stored
  } catch {
    /* fall through to a fresh id */
  }
  const id = newSessionId(secureRandom)
  try {
    sessionStorage.setItem(SESSION_STORAGE_KEY, id)
  } catch {
    /* the chat still works, the server just cannot remember it across a reload */
  }
  return id
}

/**
 * Chat state for the Ask page. Answers arrive over a Server-Sent Events stream, so the steps of the research agent show up
 * live. One question at a time; a request can be cancelled (the server then stops spending model calls); the conversation is
 * kept in sessionStorage (this tab only); a rate limit (429) blocks sending until Retry-After has passed.
 */
export function useChat() {
  const [state, dispatch] = useReducer(chatReducer, undefined, loadInitial)
  const stateRef = useRef(state)
  stateRef.current = state
  const controllers = useRef(new Map<string, AbortController>())
  const sessionId = useRef<string>('')
  if (!sessionId.current) sessionId.current = loadSessionId()
  const [blockedUntil, setBlockedUntil] = useState<number | null>(null)
  const secondsBlocked = useCountdown(blockedUntil)

  useEffect(() => {
    if (blockedUntil !== null && secondsBlocked === 0) setBlockedUntil(null)
  }, [blockedUntil, secondsBlocked])

  useEffect(() => {
    try {
      sessionStorage.setItem(CHAT_STORAGE_KEY, serializeMessages(state.messages))
    } catch {
      /* storage full or blocked: the chat still works, it just will not survive navigation */
    }
  }, [state.messages])

  useEffect(() => {
    const active = controllers.current
    return () => active.forEach((c) => c.abort()) // leaving the page: nobody is waiting for these answers
  }, [])

  const run = useCallback((id: string, question: string) => {
    const controller = new AbortController()
    controllers.current.set(id, controller)
    let timedOut = false
    let settled = false // an answer or an error event arrived
    const timer = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, ASK_TIMEOUT_MS)
    const params = new URLSearchParams({ q: question, mode: 'auto', session_id: sessionId.current })

    api
      .stream(`/api/ask/stream?${params}`, {
        signal: controller.signal,
        onEvent: (raw) => {
          const e = parseAskEvent(raw)
          if (!e || e.type === 'done') return
          if (e.type === 'answer') {
            settled = true
            const a = e.answer
            dispatch({
              type: 'answered', id, text: a.answer, citations: a.citations, route: a.route, routeReason: a.route_reason,
              trace: fromServerTrace(a.trace), usage: { llmCalls: a.usage.llm_calls, tokens: a.usage.tokens },
            })
          } else if (e.type === 'error') {
            settled = true
            if (e.status === 429 && e.retryAfter) setBlockedUntil(Date.now() + e.retryAfter * 1000)
            dispatch({ type: 'failed', id, message: e.message })
          } else {
            dispatch({ type: 'progress', id, event: e }) // route, step_start, step_result
          }
        },
      })
      .then(() => {
        if (!settled) dispatch({ type: 'failed', id, message: 'The connection closed before an answer arrived. Please try again.' })
      })
      .catch((e: unknown) => {
        if (controller.signal.aborted && !timedOut) return // cancelled by the user or the page is gone
        if (settled) return
        if (e instanceof ApiError && e.status === 429 && e.retryAfter) setBlockedUntil(Date.now() + e.retryAfter * 1000)
        dispatch({ type: 'failed', id, message: timedOut ? 'The answer took too long. Please try again.' : describeError(e, 'ask') })
      })
      .finally(() => {
        clearTimeout(timer)
        controllers.current.delete(id)
      })
  }, [])

  /** Returns false (and does nothing) if a question is already in flight or we are rate limited. */
  const send = useCallback((question: string): boolean => {
    const q = question.trim()
    if (q.length < 3 || controllers.current.size > 0 || isPending(stateRef.current)) return false
    const assistantId = newId()
    dispatch({ type: 'send', userId: newId(), assistantId, question: q })
    run(assistantId, q)
    return true
  }, [run])

  const retry = useCallback((id: string) => {
    const m = stateRef.current.messages.find((x) => x.id === id)
    if (!m || m.role !== 'assistant' || m.status !== 'error' || controllers.current.size > 0) return
    dispatch({ type: 'retry', id })
    run(id, m.question)
  }, [run])

  /** Cancels the pending question and returns its text so it can go back into the input box. */
  const cancel = useCallback((id: string): string => {
    const m = stateRef.current.messages.find((x) => x.id === id)
    controllers.current.get(id)?.abort()
    dispatch({ type: 'cancel', id })
    return m && m.role === 'assistant' ? m.question : ''
  }, [])

  /** Empties the conversation and tells the server to forget what it remembered for follow-ups. */
  const clear = useCallback(() => {
    controllers.current.forEach((c) => c.abort())
    dispatch({ type: 'clear' })
    const old = sessionId.current
    sessionId.current = newSessionId(secureRandom)
    try {
      sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId.current)
    } catch {
      /* ignore */
    }
    api.request(`/api/ask/session/${old}`, { method: 'DELETE' }).catch(() => undefined) // best effort: it expires on its own anyway
  }, [])

  return { messages: state.messages, pending: isPending(state), secondsBlocked, send, retry, cancel, clear }
}
