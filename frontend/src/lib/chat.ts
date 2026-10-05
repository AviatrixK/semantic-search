import type { Citation } from '../api/types.js'
import type { AskStreamEvent, Route } from './askStream.js'
import type { TraceStep } from './trace.js'

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
  /** How the server chose to answer (known as soon as the stream starts). */
  route: Route | null
  routeReason: string | null
  /** The agent's steps: filled in live while pending, complete when done. */
  trace: TraceStep[]
  usage: { llmCalls: number; tokens: number } | null
}

export type ChatMessage = UserMessage | AssistantMessage

export interface ChatState {
  messages: ChatMessage[]
}

export interface AnswerPayload {
  text: string
  citations: Citation[]
  route: Route
  routeReason: string
  trace: TraceStep[]
  usage: { llmCalls: number; tokens: number }
}

export type ChatAction =
  | { type: 'send'; userId: string; assistantId: string; question: string }
  /** route / step_start / step_result events from the stream, while the reply is pending. */
  | { type: 'progress'; id: string; event: AskStreamEvent }
  | ({ type: 'answered'; id: string } & AnswerPayload)
  | { type: 'failed'; id: string; message: string }
  | { type: 'retry'; id: string }
  /** Asks the same question again after an answer (not while another is pending). */
  | { type: 'regenerate'; id: string }
  /** Drops a pending reply together with the question that caused it. */
  | { type: 'cancel'; id: string }
  | { type: 'clear' }

export const initialChat: ChatState = { messages: [] }

export function isPending(state: ChatState): boolean {
  return state.messages.some((m) => m.role === 'assistant' && m.status === 'pending')
}

function patch(state: ChatState, id: string, from: AssistantMessage['status'], change: (m: AssistantMessage) => Partial<AssistantMessage>): ChatState {
  let hit = false
  const messages = state.messages.map((m) => {
    if (m.id !== id || m.role !== 'assistant' || m.status !== from) return m
    hit = true
    return { ...m, ...change(m) }
  })
  return hit ? { messages } : state // unchanged state keeps React from re-rendering
}

/** Applies a stream event to the steps list. A step_result for a step we never saw start is added, not lost. */
export function applyProgress(m: AssistantMessage, e: AskStreamEvent): Partial<AssistantMessage> {
  if (e.type === 'route') return { route: e.route, routeReason: e.reason }
  if (e.type === 'step_start') {
    const exists = m.trace.some((s) => s.step === e.step && s.index === e.index)
    return exists ? {} : { trace: [...m.trace, { step: e.step, index: e.index, tool: e.tool, label: e.label, args: e.args, summary: null, latencyMs: null, status: 'running' }] }
  }
  if (e.type === 'step_result') {
    const done = { summary: e.summary, latencyMs: e.latency_ms, status: (e.error ? 'error' : 'done') as TraceStep['status'] }
    const known = m.trace.some((s) => s.step === e.step && s.index === e.index)
    return {
      trace: known
        ? m.trace.map((s) => (s.step === e.step && s.index === e.index ? { ...s, ...done } : s))
        : [...m.trace, { step: e.step, index: e.index, tool: e.tool, label: e.summary, args: {}, ...done }],
    }
  }
  return {}
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'send':
      if (isPending(state)) return state // one question at a time
      return {
        messages: [
          ...state.messages,
          { id: action.userId, role: 'user', text: action.question },
          {
            id: action.assistantId, role: 'assistant', status: 'pending', question: action.question, text: '', citations: [], error: null,
            route: null, routeReason: null, trace: [], usage: null,
          },
        ],
      }
    case 'progress':
      return patch(state, action.id, 'pending', (m) => applyProgress(m, action.event))
    case 'answered':
      return patch(state, action.id, 'pending', () => ({
        status: 'done', text: action.text, citations: action.citations, error: null, route: action.route, routeReason: action.routeReason,
        trace: action.trace, usage: action.usage,
      }))
    case 'failed':
      return patch(state, action.id, 'pending', () => ({ status: 'error', error: action.message }))
    case 'retry':
      return isPending(state) ? state : patch(state, action.id, 'error', () => ({ status: 'pending', error: null, route: null, routeReason: null, trace: [], usage: null }))
    case 'regenerate':
      return isPending(state)
        ? state
        : patch(state, action.id, 'done', () => ({ status: 'pending', text: '', citations: [], error: null, route: null, routeReason: null, trace: [], usage: null }))
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
export const SESSION_STORAGE_KEY = 'svs.chat.session'
export const MAX_STORED_MESSAGES = 30
export const INTERRUPTED = 'This question was interrupted before it was answered. Try again.'

const isStr = (v: unknown): v is string => typeof v === 'string'
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const isRoute = (v: unknown): v is Route => v === 'search' || v === 'rag' || v === 'agent'

function citationOf(v: unknown): Citation | null {
  if (typeof v !== 'object' || v === null) return null
  const c = v as Record<string, unknown>
  return isNum(c.n) && isStr(c.video_id) && isStr(c.title) && isNum(c.start_sec) && isNum(c.end_sec)
    ? { n: c.n, video_id: c.video_id, title: c.title, start_sec: c.start_sec, end_sec: c.end_sec }
    : null
}

function stepOf(v: unknown, interrupted: boolean): TraceStep | null {
  if (typeof v !== 'object' || v === null) return null
  const s = v as Record<string, unknown>
  if (!isNum(s.step) || !isNum(s.index) || !isStr(s.tool) || !isStr(s.label)) return null
  const args = typeof s.args === 'object' && s.args !== null && !Array.isArray(s.args) ? (s.args as Record<string, unknown>) : {}
  const status = s.status === 'done' || s.status === 'error' || s.status === 'running' ? s.status : 'done'
  // A step that was still running when the page went away never finished.
  return {
    step: s.step, index: s.index, tool: s.tool, label: s.label, args, summary: isStr(s.summary) ? s.summary : interrupted && status === 'running' ? 'Interrupted' : null,
    latencyMs: isNum(s.latencyMs) ? s.latencyMs : null, status: interrupted && status === 'running' ? 'error' : status,
  }
}

function messageOf(v: unknown): ChatMessage | null {
  if (typeof v !== 'object' || v === null) return null
  const m = v as Record<string, unknown>
  if (!isStr(m.id)) return null
  if (m.role === 'user') return isStr(m.text) ? { id: m.id, role: 'user', text: m.text } : null
  if (m.role !== 'assistant' || !isStr(m.question) || !isStr(m.text)) return null
  const interrupted = m.status === 'pending'
  const citations = Array.isArray(m.citations) ? m.citations.map(citationOf).filter((c): c is Citation => c !== null) : []
  const trace = Array.isArray(m.trace) ? m.trace.map((s) => stepOf(s, interrupted)).filter((s): s is TraceStep => s !== null) : []
  const route = isRoute(m.route) ? m.route : null
  const routeReason = isStr(m.routeReason) ? m.routeReason : null
  const u = m.usage as Record<string, unknown> | null | undefined
  const usage = u && isNum(u.llmCalls) && isNum(u.tokens) ? { llmCalls: u.llmCalls, tokens: u.tokens } : null
  const base = { id: m.id, role: 'assistant' as const, question: m.question, route, routeReason }
  if (m.status === 'done') return { ...base, status: 'done', text: m.text, citations, error: null, trace, usage }
  if (m.status === 'error' && isStr(m.error)) return { ...base, status: 'error', text: '', citations: [], error: m.error, trace, usage: null }
  if (interrupted) {
    // The request died with the page: show it as failed so it can be retried.
    return { ...base, status: 'error', text: '', citations: [], error: INTERRUPTED, trace, usage: null }
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

// ---- the chat session id the server uses to remember the last turns (8-64 characters of A-Z a-z 0-9 _ -)
const ID_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

export function newSessionId(random: () => number = Math.random, length = 24): string {
  return Array.from({ length }, () => ID_ALPHABET[Math.floor(random() * ID_ALPHABET.length)]).join('')
}

export function isValidSessionId(id: unknown): id is string {
  return typeof id === 'string' && /^[A-Za-z0-9_-]{8,64}$/.test(id)
}
