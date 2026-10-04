import type { AskOut } from '../api/types.js'
import type { SseEvent } from './sse.js'

export type Route = 'search' | 'rag' | 'agent'

export type AskStreamEvent =
  | { type: 'route'; route: Route; requested: string; reason: string }
  | { type: 'step_start'; step: number; index: number; tool: string; label: string; args: Record<string, unknown> }
  | { type: 'step_result'; step: number; index: number; tool: string; summary: string; latency_ms: number; error: boolean }
  | { type: 'answer'; answer: AskOut }
  | { type: 'error'; message: string; status?: number; retryAfter?: number }
  | { type: 'done' }

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const isStr = (v: unknown): v is string => typeof v === 'string'
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const isRoute = (v: unknown): v is Route => v === 'search' || v === 'rag' || v === 'agent'

/** Turns one SSE event into a typed one, or null for anything unknown or malformed (the stream must never crash the page). */
export function parseAskEvent(e: SseEvent): AskStreamEvent | null {
  let d: unknown
  try {
    d = JSON.parse(e.data)
  } catch {
    return null
  }
  if (!isObj(d)) return null
  switch (e.event) {
    case 'route':
      return isRoute(d.route) ? { type: 'route', route: d.route, requested: isStr(d.requested) ? d.requested : '', reason: isStr(d.reason) ? d.reason : '' } : null
    case 'step_start':
      return isNum(d.step) && isNum(d.index) && isStr(d.tool) && isStr(d.label)
        ? { type: 'step_start', step: d.step, index: d.index, tool: d.tool, label: d.label, args: isObj(d.args) ? d.args : {} }
        : null
    case 'step_result':
      return isNum(d.step) && isNum(d.index) && isStr(d.tool) && isStr(d.summary)
        ? { type: 'step_result', step: d.step, index: d.index, tool: d.tool, summary: d.summary, latency_ms: isNum(d.latency_ms) ? d.latency_ms : 0, error: d.error === true }
        : null
    case 'answer':
      return isStr(d.answer) && Array.isArray(d.citations) && isRoute(d.route) ? { type: 'answer', answer: d as unknown as AskOut } : null
    case 'error':
      return isStr(d.message)
        ? { type: 'error', message: d.message, status: isNum(d.status) ? d.status : undefined, retryAfter: isNum(d.retry_after) ? d.retry_after : undefined }
        : null
    case 'done':
      return { type: 'done' }
    default:
      return null
  }
}
