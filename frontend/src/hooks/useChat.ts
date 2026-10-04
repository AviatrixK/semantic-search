import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { api } from '../api'
import type { AskOut } from '../api'
import { ApiError, describeError } from '../api/errors'
import { CHAT_STORAGE_KEY, chatReducer, isPending, restoreMessages, serializeMessages } from '../lib/chat'
import type { ChatState } from '../lib/chat'
import { useCountdown } from './useCountdown'

/** The backend allows up to 3 model calls of 30 s each (retries on 429/5xx); give up a little after that. */
const ASK_TIMEOUT_MS = 100_000
let counter = 0
const newId = () => `m${Date.now().toString(36)}-${++counter}`

function loadInitial(): ChatState {
  try {
    return { messages: restoreMessages(sessionStorage.getItem(CHAT_STORAGE_KEY)) }
  } catch {
    return { messages: [] } // storage blocked (private mode, disabled cookies)
  }
}

/**
 * Chat state for the Ask page. One question at a time; a request can be cancelled; the conversation is kept in
 * sessionStorage (this tab only) so opening a video and coming back does not lose it; a rate limit (429) blocks
 * sending until Retry-After has passed.
 */
export function useChat() {
  const [state, dispatch] = useReducer(chatReducer, undefined, loadInitial)
  const stateRef = useRef(state)
  stateRef.current = state
  const controllers = useRef(new Map<string, AbortController>())
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
    const timer = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, ASK_TIMEOUT_MS)
    api
      .request<AskOut>('/api/ask', { method: 'POST', json: { question }, signal: controller.signal })
      .then((res) => dispatch({ type: 'answered', id, text: res.answer, citations: res.citations }))
      .catch((e: unknown) => {
        if (controller.signal.aborted && !timedOut) return // cancelled by the user or the page is gone
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

  const clear = useCallback(() => {
    controllers.current.forEach((c) => c.abort())
    dispatch({ type: 'clear' })
  }, [])

  return { messages: state.messages, pending: isPending(state), secondsBlocked, send, retry, cancel, clear }
}
