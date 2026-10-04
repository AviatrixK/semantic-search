import type { Citation } from '../api/types.js'

export interface UserMessage {
  id: string
  role: 'user'
  text: string
}

export interface AssistantMessage {
  id: string
  role: 'assistant'
  status: 'pending' | 'done' | 'error'
  /** The question this reply belongs to (used by "Try again"). */
  question: string
  text: string
  citations: Citation[]
  error: string | null
}

export type ChatMessage = UserMessage | AssistantMessage

export interface ChatState {
  messages: ChatMessage[]
}

export type ChatAction =
  | { type: 'send'; userId: string; assistantId: string; question: string }
  | { type: 'answered'; id: string; text: string; citations: Citation[] }
  | { type: 'failed'; id: string; message: string }
  | { type: 'retry'; id: string }
  /** Drops a pending reply together with the question that caused it. */
  | { type: 'cancel'; id: string }
  | { type: 'clear' }

export const initialChat: ChatState = { messages: [] }

export function isPending(state: ChatState): boolean {
  return state.messages.some((m) => m.role === 'assistant' && m.status === 'pending')
}

function patch(state: ChatState, id: string, from: AssistantMessage['status'], change: Partial<AssistantMessage>): ChatState {
  let hit = false
  const messages = state.messages.map((m) => {
    if (m.id !== id || m.role !== 'assistant' || m.status !== from) return m
    hit = true
    return { ...m, ...change }
  })
  return hit ? { messages } : state // unchanged state keeps React from re-rendering
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'send':
      if (isPending(state)) return state // one question at a time
      return {
        messages: [
          ...state.messages,
          { id: action.userId, role: 'user', text: action.question },
          { id: action.assistantId, role: 'assistant', status: 'pending', question: action.question, text: '', citations: [], error: null },
        ],
      }
    case 'answered':
      return patch(state, action.id, 'pending', { status: 'done', text: action.text, citations: action.citations, error: null })
    case 'failed':
      return patch(state, action.id, 'pending', { status: 'error', error: action.message })
    case 'retry':
      return isPending(state) ? state : patch(state, action.id, 'error', { status: 'pending', error: null })
    case 'cancel': {
      const i = state.messages.findIndex((m) => m.id === action.id && m.role === 'assistant' && m.status === 'pending')
      if (i < 0) return state
      const drop = new Set([action.id, state.messages[i - 1]?.role === 'user' ? state.messages[i - 1].id : ''])
      return { messages: state.messages.filter((m) => !drop.has(m.id)) }
    }
    case 'clear':
      return state.messages.length ? initialChat : state
  }
}

// ---- keeping the conversation across navigation (sessionStorage: per tab, gone when the tab closes)
export const CHAT_STORAGE_KEY = 'svs.chat.v1'
export const MAX_STORED_MESSAGES = 30
export const INTERRUPTED = 'This question was interrupted before it was answered. Try again.'

const isStr = (v: unknown): v is string => typeof v === 'string'
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

function citationOf(v: unknown): Citation | null {
  if (typeof v !== 'object' || v === null) return null
  const c = v as Record<string, unknown>
  return isNum(c.n) && isStr(c.video_id) && isStr(c.title) && isNum(c.start_sec) && isNum(c.end_sec)
    ? { n: c.n, video_id: c.video_id, title: c.title, start_sec: c.start_sec, end_sec: c.end_sec }
    : null
}

function messageOf(v: unknown): ChatMessage | null {
  if (typeof v !== 'object' || v === null) return null
  const m = v as Record<string, unknown>
  if (!isStr(m.id)) return null
  if (m.role === 'user') return isStr(m.text) ? { id: m.id, role: 'user', text: m.text } : null
  if (m.role !== 'assistant' || !isStr(m.question) || !isStr(m.text)) return null
  const citations = Array.isArray(m.citations) ? m.citations.map(citationOf).filter((c): c is Citation => c !== null) : []
  if (m.status === 'done') return { id: m.id, role: 'assistant', status: 'done', question: m.question, text: m.text, citations, error: null }
  if (m.status === 'error' && isStr(m.error)) return { id: m.id, role: 'assistant', status: 'error', question: m.question, text: '', citations: [], error: m.error }
  if (m.status === 'pending') {
    // The request died with the page: show it as failed so it can be retried.
    return { id: m.id, role: 'assistant', status: 'error', question: m.question, text: '', citations: [], error: INTERRUPTED }
  }
  return null
}

export function serializeMessages(messages: readonly ChatMessage[]): string {
  return JSON.stringify(messages.slice(-MAX_STORED_MESSAGES))
}

/** Rebuilds a conversation from storage; anything malformed is dropped rather than trusted. */
export function restoreMessages(raw: string | null): ChatMessage[] {
  if (!raw) return []
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.map(messageOf).filter((m): m is ChatMessage => m !== null).slice(-MAX_STORED_MESSAGES)
  } catch {
    return []
  }
}
