import type { TraceStepOut } from '../api/types.js'
import type { Route } from './askStream.js'

export interface TraceStep {
  step: number
  index: number
  tool: string
  /** "Searching “x”": shown while the call runs. */
  label: string
  args: Record<string, unknown>
  /** "Found 5 clips": shown when it is done. */
  summary: string | null
  latencyMs: number | null
  status: 'running' | 'done' | 'error'
}

export function formatLatency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return ''
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

const ROUTES: Record<Route, { label: string; hint: string }> = {
  search: { label: 'Search results', hint: 'Shown as a search: no AI model was used' },
  rag: { label: 'Direct answer', hint: 'Answered from one search, with a single AI call' },
  agent: { label: 'Research agent', hint: 'Planned its own searches and read transcripts' },
}

export function routeLabel(route: Route | null): { label: string; hint: string } | null {
  return route ? ROUTES[route] : null
}

/** One-line text for a step in the list: the label while it runs, the outcome when it is done. */
export function stepLine(s: TraceStep): string {
  if (s.status === 'running') return `${s.label}…`
  return s.summary ?? s.label
}

/** Parallel calls of one round are shown together: group by round. */
export function groupByStep(steps: readonly TraceStep[]): TraceStep[][] {
  const groups: TraceStep[][] = []
  for (const s of steps) {
    const last = groups.at(-1)
    if (last && last[0].step === s.step) last.push(s)
    else groups.push([s])
  }
  return groups
}

/** Readable arguments for the details view: `query: "x"`, one per line. */
export function formatArgs(args: Record<string, unknown>): string {
  const lines = Object.entries(args).map(([k, v]) => `${k}: ${typeof v === 'string' ? JSON.stringify(v) : String(v)}`)
  return lines.length ? lines.join('\n') : '(no arguments)'
}

/** The finished trace the server returns with an answer, as UI steps. */
export function fromServerTrace(steps: readonly TraceStepOut[]): TraceStep[] {
  return steps.map((s) => ({
    step: s.step, index: s.index, tool: s.tool, label: s.label, args: s.args, summary: s.summary, latencyMs: s.latency_ms,
    status: s.error ? 'error' : 'done',
  }))
}
